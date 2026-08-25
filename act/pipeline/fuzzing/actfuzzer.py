"""
ACTFuzzer: Inference-based whitebox fuzzing for neural network verification.

Main fuzzer engine that orchestrates mutation, coverage tracking, and
property checking to find counterexamples.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, cast
import os
import time
import json
import torch
import torch.nn as nn
from pathlib import Path

from act.front_end.specs import InputSpec, OutputSpec
from act.front_end.spec_creator_base import LabeledInputTensor
from act.front_end.verifiable_model import (
    InputLayer,
    InputSpecLayer,
    OutputSpecLayer,
    VerifiableModel,
)
from act.pipeline.fuzzing.mutations import MutationEngine, _relu_sign_pattern_batched
from act.pipeline.fuzzing.coverage import CoverageTracker
from act.pipeline.fuzzing.corpus import SeedCorpus, FuzzingSeed
from act.pipeline.fuzzing.checker import Counterexample, PropertyChecker
from act.util.path_config import get_pipeline_log_dir, get_project_root
from act.util.device_manager import get_default_device
from act.util.format_utils import rule


@dataclass
class FuzzingConfig:
    """
    Fuzzing configuration (immutable).

    Attributes:
        max_iterations: Maximum fuzzing iterations
        timeout_seconds: Total time budget
        seed_selection_strategy: "energy" or "random"
        mutation_weights: Dict of strategy weights
        coverage_strategy: Coverage tracking strategy ("BestInputCov" or "GlobalCov")
        activation_threshold: Neuron activation threshold for coverage tracking
        perturb_mode: Perturbation size computation mode ("adaptive_scalar", "adaptive_perdim", "fixed")
        perturb_scale: Fraction of range per mutation perturbation (e.g., 0.1 = 10% = ~10 steps to traverse)
        save_counterexamples: Whether to save counterexamples incrementally
        output_dir: Output directory for results
        report_interval: Print progress every N iterations
        verbose: Logging verbosity (0=silent, 1=report violations in progress only, 2=print each violation immediately)
        trace_level: Execution tracing level (0=disabled, 1=default, 2=full, 3=debug)
        trace_sample_rate: Capture every Nth iteration (1=all iterations)
        trace_storage: Storage backend ("hdf5" or "json")
        trace_output: Trace output path (None=auto-generate)

    Device Management:
        Device is controlled by act.util.device_manager (the single source of truth for the entire ACT pipeline).

    Perturbation Size Configuration:
        NOTE: We use "perturb_size" (not "epsilon") to avoid confusion with InputSpec.eps (L∞ radius).
        - InputSpec.eps: Defines constraint boundaries (e.g., center ± eps for LINF_BALL)
        - Mutation perturb_size: Controls mutation perturbation magnitude (exploration granularity)

        perturb_mode determines how mutation perturbation sizes are computed:
        - "adaptive_scalar": Single perturb_size from mean(ub-lb) * perturb_scale (default, best for uniform ranges)
        - "adaptive_perdim": Per-dimension perturb_size from (ub-lb) * perturb_scale (best for non-uniform ranges)
        - "fixed": Legacy hardcoded values (0.01 for gradient/activation, 0.005 for boundary/random)

        coverage_strategy determines the coverage strategy to use:
        - "BestInputCov": Per-input coverage (best per-input coverage over time)
        - "GlobalCov": Global union coverage (monotonic union over all inputs)

        perturb_scale interpretation:
        - Fraction of feasible range each mutation perturbation covers
        - steps_to_traverse = 1 / perturb_scale
        - Example: perturb_scale=0.1 → 10% per perturbation → ~10 steps to traverse from lb to ub
    """

    # All configuration values are loaded from config.yaml via from_yaml().
    # from_yaml() is the single source of configuration truth.
    max_iterations: int
    timeout_seconds: float
    seed_selection_strategy: str
    mutation_weights: Dict[str, float]
    coverage_strategy: str
    activation_threshold: float
    perturb_mode: str
    perturb_scale: float
    save_counterexamples: bool
    output_dir: Path
    report_interval: int
    verbose: int

    # Tracing configuration
    trace_level: int
    trace_sample_rate: int
    trace_storage: str
    trace_output: Optional[Path]

    # Stop as soon as the first counterexample is found (for a fast pre-attack:
    # total_time then measures time-to-first-counterexample). Default off.
    stop_on_first_violation: bool = False

    # -- PatternStateManager (global BK-tree/Bloom-filter state tracking) --
    # admission_mode: "coverage" (default, original behavior unchanged) uses
    #   CoverageTracker's neuron-activation interestingness for every
    #   strategy, exactly as before this feature existed.
    #   "state" replaces that, for every strategy, with PatternStateManager's
    #   novelty/diversity check over the unstable-ReLU subspace (energy 10 on
    #   admission, same as coverage-interesting did).
    admission_mode: str = "coverage"
    # scheduling_mode: "energy" (default, original behavior) keeps
    #   SeedCorpus's own energy-weighted select(). "sparse" instead draws
    #   seeds from PatternStateManager.pick_seeds() (density-weighted,
    #   optionally blended with each seed's energy_bonus).
    scheduling_mode: str = "energy"
    # BI (explore, density-guided) / GCE (exploit, anchor pull-back near a
    # known counterexample) two-task loop, sharing one PatternStateManager.
    # Off by default -- when off, fuzz() is the original single-phase loop.
    enable_bi_gce: bool = False
    bi_batch_size: int = 0   # 0 = use the normal (model-synthesis) batch size
    gce_batch_size: int = 0  # 0 = same as bi_batch_size
    # PatternStateManager tuning (only used when admission_mode=="state",
    # scheduling_mode=="sparse", or enable_bi_gce=True).
    state_diversity_threshold: int = 1
    state_bloom_bits: int = 1 << 20
    state_bloom_hashes: int = 4
    state_local_bias_high: float = 10.0
    state_local_bias_low: float = 0.1
    # HPGD tuning (the "hpgd" MutationEngine strategy).
    hpgd_flip_count: int = 10
    hpgd_num_steps: int = 10
    hpgd_margin: float = 0.01
    # Flip the target neurons one at a time (each step re-reads the pattern
    # actually reached) instead of committing to all hpgd_flip_count flips up front.
    hpgd_sequential_flip: bool = False
    # Which neurons the hinge loss sums over: "target_only" scores just the
    # flipped targets; other scopes also penalize drift in the untouched ones.
    hpgd_loss_scope: str = "target_only"
    # >0 steps on a running momentum buffer's sign instead of the raw gradient's.
    hpgd_momentum: float = 0.0
    # Halve step_size at Auto-PGD's checkpoint fractions.
    hpgd_step_decay: bool = False
    # Down-weights the "hold still" neurons (those whose target equals their
    # natural sign) relative to the ones being actively flipped. 1.0 = equal.
    hpgd_hold_still_weight: float = 1.0
    # Normalize each neuron's hinge term by its activation scale, so neurons
    # with large pre-activations don't dominate the summed loss.
    hpgd_normalize_by_scale: bool = False
    # HPGD-Cov (the "hpgd_cov" strategy) tuning: coverage-targeted, each sample
    # chases its own randomly-drawn never-activated neurons.
    hpgd_cov_target_count: int = 3
    hpgd_cov_num_steps: int = 10
    hpgd_cov_momentum: float = 0.0
    hpgd_cov_step_decay: bool = False
    # Chase the single uncovered neuron closest to the firing threshold from
    # below (easiest to flip) instead of hpgd_cov_target_count random ones.
    hpgd_cov_nearest_margin: bool = False
    # GCE (HPGDPullbackMutation) tuning.
    gce_num_steps: int = 10
    gce_margin: float = 0.01
    gce_hamming_radius: int = 3
    gce_noise_scale: float = 0.05
    gce_step_decay: bool = False

    # -- BI (broad, sparsity-guided explore) phase --
    # None = BI runs MutationEngine's weighted portfolio. A strategy name here
    # makes _fuzz_iteration dispatch to that attack ALONE, independent of
    # enable_bi_gce/admission_mode/scheduling_mode.
    bi_attack_strategy: Optional[str] = None
    bi_attack_pgd_steps: int = 50
    bi_attack_apgd_t_target_classes: int = 5
    # Probability of restarting a BI batch from fresh random points in the box
    # rather than from corpus seeds; multiplied by the cooling rate each time
    # it fires, so restarts thin out as the run progresses. 0 = never.
    bi_random_restart_prob: float = 0.0
    bi_random_restart_cooling_rate: float = 0.98
    # Run BI's HPGD and PGD stages as a producer/consumer thread pair (see
    # act/pipeline/fuzzing/bi_threads.py) instead of sequentially.
    bi_threaded: bool = False
    bi_queue_size: int = 64

    # Tensor dtype for the pipeline tier. See act/config/pipeline.yaml for why
    # this deliberately differs from the back_end tier.
    dtype: str = "float32"

    # Independent PGD random starts per mutation; the best lane-wise result
    # wins and restarts stop early once every lane violates. 1 = single start,
    # i.e. no extra cost.
    pgd_restarts: int = 1

    # Used instead of pgd_restarts once sign estimators are installed, which
    # only happens on a binarized network. Restarts alternate the estimator
    # between its loose and tight eps.
    pgd_restarts_binarized: int = 40

    def __post_init__(self):
        """Normalize output_dir to Path object."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(get_pipeline_log_dir()) / self.output_dir
        elif not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)

    @classmethod
    def from_yaml(
        cls, config_path: Optional[str | Path] = None, **overrides
    ) -> "FuzzingConfig":
        """Load FuzzingConfig from the pipeline YAML with optional overrides.

        The YAML file is read by act.config.config (the single reader of the
        config YAML files); this only merges overrides and constructs.
        """
        from act.config.config import read_fuzzing_section

        return cls.from_mapping(read_fuzzing_section(config_path), **overrides)

    @classmethod
    def from_mapping(cls, section: Dict[str, Any], **overrides) -> "FuzzingConfig":
        """Build from an already-parsed ``fuzzing`` mapping (no file I/O)."""
        merged_config = {**section, **overrides}
        if "output_dir" in merged_config and isinstance(
            merged_config["output_dir"], str
        ):
            merged_config["output_dir"] = (
                Path(get_pipeline_log_dir()) / merged_config["output_dir"]
            )
        return cls(**merged_config)


@dataclass
class FuzzingReport:
    """
    Fuzzing results summary.

    Attributes:
        total_iterations: Number of iterations completed
        total_time: Time elapsed in seconds
        counterexamples: List of found counterexamples
        neuron_coverage: Final neuron coverage (0.0 to 1.0)
        total_mutations: Total mutations applied
        seeds_explored: Number of unique seeds explored
        num_of_never_activated_neurons: Number of neurons that were never activated across all iterations
        never_activated_neurons: Sample of never-activated neuron ids (layer_name, neuron_idx)
    """

    total_iterations: int
    total_time: float
    counterexamples: List[Counterexample]
    neuron_coverage: float
    total_mutations: int
    seeds_explored: int
    num_of_never_activated_neurons: int = 0
    never_activated_neurons: List[Tuple[str, int]] = field(default_factory=list)

    def save(self, output_dir: Path):
        """Save report and counterexamples to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary as JSON
        summary = {
            "iterations": self.total_iterations,
            "time_seconds": self.total_time,
            "counterexamples_found": len(self.counterexamples),
            "neuron_coverage": self.neuron_coverage,
            "mutations": self.total_mutations,
            "seeds_explored": self.seeds_explored,
            "num_of_never_activated_neurons": self.num_of_never_activated_neurons,
            # JSON-friendly: list of [layer_name, neuron_idx]
            "never_activated_neurons": [
                [ln, int(i)] for (ln, i) in self.never_activated_neurons
            ],
        }

        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save counterexamples
        for i, ce in enumerate(self.counterexamples):
            ce.save(output_dir / f"counterexample_{i}.pt")

        print(f"✅ Report saved to {os.path.relpath(output_dir)}")


class ACTFuzzer:
    """
    Inference-based whitebox fuzzer for neural network verification.

    Features:
    - Gradient-guided mutations (FGSM-style)
    - Neuron coverage tracking (DeepXplore)
    - Energy-based seed scheduling (AFL)
    - OutputSpec violation detection
    - InputSpec constraint projection

    Workflow:
    1. Initialize with wrapped model and seeds
    2. Loop: Select seed → Mutate → Inference → Check violation → Update coverage
    3. Return report with counterexamples

    Example:
        >>> config = FuzzingConfig.from_yaml(max_iterations=5000)
        >>> fuzzer = ACTFuzzer(
        ...     wrapped_model=model,
        ...     initial_seeds=labeled_tensors,
        ...     config=config
        ... )
        >>> report = fuzzer.fuzz()
        >>> print(f"Found {len(report.counterexamples)} violations")
    """

    def __init__(
        self,
        wrapped_model: nn.Module,
        initial_seeds: List[LabeledInputTensor],
        config: Optional[FuzzingConfig] = None,
    ):
        """
        Initialize ACTFuzzer.

        Args:
            wrapped_model: VerifiableModel from model_synthesis.
                          Contains InputSpecLayer and OutputSpecLayer with batched specs
                          sized for N VNNLib instances.
            initial_seeds: List of LabeledInputTensor from spec creators
            config: Fuzzing configuration (uses from_yaml() defaults if None)

        Initialization Steps:
            1. Load config → from_yaml() or provided FuzzingConfig
            2. Get device from device_manager → get_default_device()
            3. Use wrapped model directly → self.model (no spec layer stripping)
            4. Extract specs → _extract_spec(InputSpecLayer), _extract_spec(OutputSpecLayer)
            5. Determine batch size → from InputSpec bounds shape[0] (model synthesis N)
            6. Initialize components → MutationEngine, CoverageTracker, PropertyChecker, SeedCorpus
            7. Setup tracer → ExecutionTracer (if trace_level > 0)
        """
        self.config = config or FuzzingConfig.from_yaml()
        self.device = get_default_device()

        assert not VerifiableModel.get_strict_mode(), (
            "ACTFuzzer requires VerifiableModel strict mode to be disabled; "
            "violations must be returned to PropertyChecker, not raised by forward()"
        )
        self.model = wrapped_model.to(self.device)

        # Extract specs for MutationEngine (projection) and PropertyChecker (violation detection).
        self.input_spec = cast(Optional[InputSpec], self._extract_spec(InputSpecLayer))
        self.output_spec = cast(Optional[OutputSpec], self._extract_spec(OutputSpecLayer))

        input_layer = next(
            (layer for layer in self.model.children() if isinstance(layer, InputLayer)),
            None,
        )
        assert input_layer is not None, "ACTFuzzer requires an InputLayer"
        assert initial_seeds, "ACTFuzzer requires at least one initial seed"
        assert all(seed.tensor.shape[0] == 1 for seed in initial_seeds), (
            "Each initial_seeds[i] must represent exactly one spec row"
        )
        seed_inputs = torch.cat([seed.tensor for seed in initial_seeds], dim=0).to(
            device=input_layer.input_tensor.device,
            dtype=input_layer.input_tensor.dtype,
        )
        assert seed_inputs.shape == input_layer.input_tensor.shape, (
            "Initial seed count/shape must match synthesized spec rows: "
            f"seeds={tuple(seed_inputs.shape)}, specs={tuple(input_layer.input_tensor.shape)}"
        )
        assert torch.equal(seed_inputs, input_layer.input_tensor), (
            "initial_seeds[i] must equal InputLayer row i; original_index relies "
            "on this seed/spec-row alignment"
        )

        # Batch size is determined by model synthesis (number of VNNLib instances).
        self.batch_size = (
            self.input_spec.lb.shape[0]
            if self.input_spec and self.input_spec.lb is not None
            else len(initial_seeds)
        )

        # Initialize components
        self.mutation_engine = MutationEngine(
            model=self.model,
            input_spec=self.input_spec,
            weights=self.config.mutation_weights,
            perturb_mode=self.config.perturb_mode,
            perturb_scale=self.config.perturb_scale,
            pgd_restarts=self.config.pgd_restarts,
            pgd_restarts_binarized=self.config.pgd_restarts_binarized,
        )
        self.coverage_tracker = CoverageTracker(
            model=self.model,
            threshold=self.config.activation_threshold,
            strategy=self.config.coverage_strategy,
        )

        # HPGD-Cov optimizes the tracker's own neuron space, so it can only be
        # built once the tracker exists. Its margin is pinned to the tracker's
        # firing threshold: a different value would optimize a different
        # condition than the one coverage reporting measures.
        _hpgd_cov = self.mutation_engine.strategies.get("hpgd_cov")
        if _hpgd_cov is not None:
            _hpgd_cov.coverage_tracker = self.coverage_tracker
            _hpgd_cov.margin = float(self.config.activation_threshold)
            _hpgd_cov.target_count = int(self.config.hpgd_cov_target_count)
            _hpgd_cov.num_steps = int(self.config.hpgd_cov_num_steps)
            _hpgd_cov.momentum = float(self.config.hpgd_cov_momentum)
            _hpgd_cov.step_decay = bool(self.config.hpgd_cov_step_decay)
            _hpgd_cov.nearest_margin = bool(self.config.hpgd_cov_nearest_margin)

        self.property_checker = PropertyChecker(self.output_spec)
        self.seed_corpus = SeedCorpus(
            initial_seeds=initial_seeds, strategy=self.config.seed_selection_strategy
        )
        assert len(self.seed_corpus) == self.batch_size, (
            "Initial corpus must preserve exactly one seed slot per synthesized "
            f"spec row: corpus={len(self.seed_corpus)}, specs={self.batch_size}"
        )

        # PatternStateManager: only built when actually needed (admission_mode
        # "state", scheduling_mode "sparse", or BI/GCE), so the default
        # "coverage" + "energy" configuration pays zero extra cost and is
        # byte-for-byte the original fuzzer.
        self.state_manager = None
        self.gce_mutation = None
        needs_state_manager = (
            self.config.admission_mode == "state"
            or self.config.scheduling_mode == "sparse"
            or self.config.enable_bi_gce
        )
        if needs_state_manager:
            self._init_state_manager(initial_seeds)

        # Initialize tracer (only if trace_level > 0)
        if self.config.trace_level > 0:
            from act.pipeline.fuzzing.tracer import ExecutionTracer

            # Auto-generate trace output path if not specified
            # Class-level counter for unique trace filenames across multiple ACTFuzzer instances.
            # When fuzzing multiple VNNLib instances (one ACTFuzzer per instance), each needs a
            # distinct trace file (traces_0.json, traces_1.json, ...) to avoid overwriting.
            if self.config.trace_output is not None:
                trace_output = self.config.trace_output
            else:
                if not hasattr(ACTFuzzer, "_trace_counter"):
                    ACTFuzzer._trace_counter = 0
                ext = self._get_trace_ext()
                trace_output = (
                    self.config.output_dir / f"traces_{ACTFuzzer._trace_counter}.{ext}"
                )
                ACTFuzzer._trace_counter += 1

            self.tracer = ExecutionTracer(
                level=self.config.trace_level,
                sample_rate=self.config.trace_sample_rate,
                storage_backend=self.config.trace_storage,
                output_path=trace_output,
            )

            print(
                f"📊 Tracing enabled: Level {self.config.trace_level}, "
                f"sampling every {self.config.trace_sample_rate} iteration(s)"
            )
            print(f"   Output: {os.path.relpath(trace_output)}")
        else:
            self.tracer = None  # No overhead when disabled

        # Statistics
        self.counterexamples: List[Counterexample] = []
        self.iterations = 0
        self.start_time = 0.0
        self.never_activated_neurons: List[Tuple[str, int]] = []
        self.last_report_ce_count = 0  # Track counterexamples count at last report

    def _get_trace_ext(self) -> str:
        """Get file extension for trace storage."""
        return {"hdf5": "h5", "json": "json"}[self.config.trace_storage]

    def _extract_spec(self, layer_type) -> Optional[InputSpec | OutputSpec]:
        """Extract spec from wrapper layer by type."""
        for layer in self.model.children():
            if isinstance(layer, layer_type):
                return cast(InputSpec | OutputSpec, cast(object, layer.spec))
        return None

    def _init_state_manager(self, initial_seeds: List[LabeledInputTensor]) -> None:
        """Build PatternStateManager: compute the instance's unstable-ReLU
        mask (via act.back_end interval bound propagation, falling back to
        "every neuron is a candidate" if that fails for any reason -- e.g.
        an unsupported layer type), wire it into the "hpgd" strategy's flip
        candidates, and construct the GCE pull-back mutation."""
        from act.pipeline.fuzzing.mutations import HPGDPullbackMutation, _relu_preactivations_batched
        from act.pipeline.fuzzing.state_manager import PatternStateManager, compute_unstable_mask

        sample = initial_seeds[0].tensor.to(self.device)
        with torch.no_grad():
            total_neurons = int(_relu_preactivations_batched(self.model, sample).shape[1])

        unstable_mask = None
        if self.input_spec is not None:
            try:
                lb, ub = self.input_spec.materialize_box_seed()
                unstable_mask = compute_unstable_mask(self.model, lb.to(self.device)[:1], ub.to(self.device)[:1])
            except Exception:
                unstable_mask = None

        self.state_manager = PatternStateManager(
            unstable_mask=unstable_mask,
            total_neurons=total_neurons,
            diversity_threshold=self.config.state_diversity_threshold,
            bloom_bits=self.config.state_bloom_bits,
            bloom_hashes=self.config.state_bloom_hashes,
            local_bias_high=self.config.state_local_bias_high,
            local_bias_low=self.config.state_local_bias_low,
            device=self.device,
        )

        hpgd = self.mutation_engine.strategies.get("hpgd")
        if hpgd is not None:
            hpgd.candidate_indices = self.state_manager.candidate_indices
            hpgd.flip_count = self.config.hpgd_flip_count
            hpgd.num_steps = self.config.hpgd_num_steps
            hpgd.margin = self.config.hpgd_margin

        if self.config.enable_bi_gce:
            self.gce_mutation = HPGDPullbackMutation(
                num_steps=self.config.gce_num_steps,
                margin=self.config.gce_margin,
                hamming_radius=self.config.gce_hamming_radius,
                noise_scale=self.config.gce_noise_scale,
            )

    def _observe_state(
        self,
        parent_seeds: "FuzzingSeed",
        child_inputs: torch.Tensor,
        natural_pattern: torch.Tensor,
        achieved_pattern: torch.Tensor,
        violation_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Feed a batch of mutation results (regardless of which strategy
        produced them) into PatternStateManager: admits each sample whose
        achieved pattern is novel/diverse enough, records it (and, for
        confirmed violations, as a GCE anchor), and updates that new seed's
        local_bias from the positions that actually flipped. Returns the
        per-sample admitted BoolTensor[B], for the caller to use as (or
        fold into) interesting_mask."""
        assert self.state_manager is not None
        B = child_inputs.shape[0]
        admitted = torch.zeros(B, dtype=torch.bool, device=child_inputs.device)
        for b in range(B):
            is_ce = bool(violation_mask[b].item())
            ok = self.state_manager.observe(
                seed_tensor=child_inputs[b : b + 1],
                pattern_full=achieved_pattern[b : b + 1],
                label=parent_seeds.label[b : b + 1],
                original_tensor=parent_seeds.original_tensor[b : b + 1],
                original_index=parent_seeds.original_index[b : b + 1],
                is_ce=is_ce,
                energy_bonus=(5.0 if is_ce else 1.0),
            )
            admitted[b] = ok
            if ok:
                restricted_natural = self.state_manager.restrict(natural_pattern[b])
                restricted_achieved = self.state_manager.restrict(achieved_pattern[b])
                flipped = (restricted_natural != restricted_achieved).nonzero(as_tuple=True)[0]
                self.state_manager.update_local_bias(child_inputs[b : b + 1], flipped)
        return admitted

    def fuzz(self) -> FuzzingReport:
        """
        Main fuzzing loop.

        Returns:
            FuzzingReport with counterexamples and statistics
        """
        print(f"{rule()}")
        print(f"ACT: Abstract Constraint Transformer")
        print(f"Inference-based whitebox fuzzing for neural network verification")
        print(f"{rule()}\n")

        batch_size = self.batch_size

        print(f"🚀 Starting ACTFuzzer with {len(self.seed_corpus)} seeds")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {batch_size} (from model synthesis)")
        print(f"   Max iterations: {self.config.max_iterations}")
        print(f"   Timeout: {self.config.timeout_seconds}s\n")

        self.start_time = time.time()
        iteration = 0

        while iteration < self.config.max_iterations:
            if time.time() - self.start_time > self.config.timeout_seconds:
                print(f"⏱️  Timeout reached after {iteration} iterations")
                break

            # Batch size normally must match VerifiableModel's spec layer dimensions
            # (InputSpecLayer bounds are sized [N, ...] from model synthesis), but
            # MutationEngine._project() gathers bounds per-sample via
            # seeds.original_index, so an independently-sized BI/GCE batch (see
            # bi_batch_size/gce_batch_size) is still projected correctly.
            bi_bs = self.config.bi_batch_size or batch_size if self.config.enable_bi_gce else batch_size
            self._fuzz_iteration(iteration, bi_bs)
            if self.config.enable_bi_gce:
                # GCE rides alongside BI's iteration, sharing the same
                # PatternStateManager/corpus; it idles until a counterexample
                # exists to anchor on. It does not consume its own iteration budget.
                gce_bs = self.config.gce_batch_size or bi_bs
                self._gce_iteration(iteration, gce_bs)
            iteration += bi_bs

            if self.config.stop_on_first_violation and self.counterexamples:
                print(f"✋ First counterexample at {time.time() - self.start_time:.3f}s; stopping")
                break

            if iteration > 0 and iteration % self.config.report_interval < batch_size:
                self._print_progress(iteration)

        return self._generate_report()

    def _fuzz_iteration(self, start_iteration: int, batch_size: int):
        """
        Run one batch-native fuzzing iteration over batch_size samples.

        All operations use FuzzingSeed batch tensors

        Args:
            start_iteration: Starting iteration number
            batch_size: Number of samples to process
        """
        # 1. select — returns FuzzingSeed batch (B=batch_size). "sparse"
        # scheduling draws density-weighted seeds from PatternStateManager
        # instead of SeedCorpus's energy-weighted select(); falls back to
        # the normal corpus (e.g. before any state has been recorded yet).
        if (
            self.config.scheduling_mode == "sparse"
            and self.state_manager is not None
            and len(self.state_manager) > 0
        ):
            payloads = self.state_manager.pick_seeds(batch_size, use_energy=True)
            seeds: FuzzingSeed = self.state_manager.seeds_to_batch(payloads)
        else:
            seeds = self.seed_corpus.select(batch_size)

        # PatternStateManager admission needs the PRE-mutation pattern to
        # later diff against the achieved one (which candidate neurons this
        # specific mutation actually flipped, for local_bias).
        natural_pattern = None
        if self.config.admission_mode == "state" and self.state_manager is not None:
            with torch.no_grad():
                natural_pattern = _relu_sign_pattern_batched(self.model, seeds.tensor.to(self.device))

        # 2. mutate — takes FuzzingSeed, returns Tensor[B, ...]
        inputs = self.mutation_engine.mutate(seeds)

        # 3. inference
        with torch.no_grad():
            output = self.model(inputs)
        outputs = output["output"] if isinstance(output, dict) else output

        # 4. violation check — returns (BoolTensor[B], List[Counterexample])
        violation_mask, counterexamples = self.property_checker.check(
            inputs=inputs,
            outputs=outputs,
            seeds=seeds,
        )

        # 5. coverage update — returns per-sample interestingness mask.
        # Always computed/accumulated regardless of strategy (see step 6),
        # even though HPGD doesn't use it to gate corpus admission.
        activations = self.mutation_engine.get_activation_map()
        global_delta, cov_interesting = self.coverage_tracker.update(
            inputs, activations
        )
        # Update secondary strategy for dual coverage reporting
        _other = "BestInputCov" if self.config.coverage_strategy == "GlobalCov" else "GlobalCov"
        self.coverage_tracker.update(inputs, activations, strategy=_other)

        # 6. energy computation (fully vectorized)
        # admission_mode "state" (any strategy, not just HPGD): a mutated
        # sample is "interesting" iff PatternStateManager judges its
        # achieved ReLU-sign pattern novel/diverse enough (Bloom filter +
        # BK-tree over the unstable subspace) -- CoverageTracker's own
        # per-neuron-ever-activated signal (cov_interesting, still computed
        # above) is not consulted. admission_mode "coverage" (default)
        # keeps the original behavior unchanged for every strategy.
        if self.config.admission_mode == "state" and self.state_manager is not None:
            with torch.no_grad():
                achieved_pattern = _relu_sign_pattern_batched(self.model, inputs)
            admitted = self._observe_state(seeds, inputs, natural_pattern, achieved_pattern, violation_mask)
            interesting_mask = violation_mask | admitted
            energies = admitted.float() * 10.0 + violation_mask.float() * 100.0
        else:
            interesting_mask = violation_mask | cov_interesting
            energies = cov_interesting.float() * 10.0 + violation_mask.float() * 100.0
        energies = torch.clamp(energies, min=0.1)

        # 7. counterexamples — already sparse list from checker
        for ce in counterexamples:
            self.counterexamples.append(ce)
            if self.config.verbose >= 2:
                print(
                    f"🚨 Counterexample #{len(self.counterexamples)}: "
                    f"{ce.summary()}"
                )
            if self.config.save_counterexamples:
                self.config.output_dir.mkdir(parents=True, exist_ok=True)
                ce.save(self.config.output_dir / f"ce_{len(self.counterexamples)}.pt")

        # 8. Corpus add — batch add with mask (no per-sample loop)
        child_seeds = FuzzingSeed(
            tensor=inputs,
            original_tensor=seeds.original_tensor,
            original_index=seeds.original_index,
            label=seeds.label,
            energy=energies,
            depth=seeds.depth + 1,
            parent_id=seeds.id,
            select_count=seeds.select_count,
        )
        self.seed_corpus.add(child_seeds, interesting_mask)

        # 9. Tracing ( per-sample for detail)
        if self.tracer:
            counterexamples_by_lane = dict(
                zip(violation_mask.nonzero(as_tuple=True)[0].tolist(), counterexamples)
            )
            coverage = self.coverage_tracker.get_coverage()
            mutation_strategy = self.mutation_engine.last_strategy or "unknown"
            gradients = None
            loss_value = None
            if self.config.trace_level >= 3:
                gradients = self.mutation_engine.get_last_gradients()
                loss_value = self.mutation_engine.get_last_loss()

            for i in range(batch_size):
                iteration = start_iteration + i
                if self.tracer.should_trace(iteration):
                    self.tracer.record_iteration(
                        iteration=iteration,
                        timestamp=time.time(),
                        mutation_strategy=mutation_strategy,
                        violation=counterexamples_by_lane.get(i),
                        coverage=coverage,
                        coverage_delta=global_delta / batch_size,
                        energy=float(energies[i]),
                        seed_id=str(int(seeds.id[i].item())),
                        input_before=seeds.tensor[i : i + 1],
                        input_after=inputs[i : i + 1],
                        parent_id=str(int(seeds.parent_id[i].item())),
                        depth=int(seeds.depth[i].item()),
                        activations=activations,
                        gradients=gradients,
                        loss_value=loss_value,
                    )

        self.iterations = start_iteration + batch_size

    def _gce_iteration(self, start_iteration: int, batch_size: int) -> None:
        """GCE (generate-counterexample) phase: HPGDPullbackMutation anchored
        on confirmed counterexamples already recorded in PatternStateManager
        (found by BI or by whichever strategy admitted them). Idles (no-op)
        until at least one counterexample exists -- mirrors
        PatternSearchPGD's round 3 only running once round 2 found a real
        counterexample. Rides alongside BI's iteration budget rather than
        consuming its own (see fuzz()); does not feed the execution tracer.
        """
        if self.state_manager is None or self.gce_mutation is None:
            return
        payloads = self.state_manager.pick_ce_seeds(batch_size)
        if not payloads:
            return

        anchors: FuzzingSeed = self.state_manager.seeds_to_batch(payloads)
        anchor_tensor = anchors.tensor.to(self.device)
        restricted_anchor_patterns = torch.stack([p["pattern"] for p in payloads], dim=0).to(self.device)

        with torch.no_grad():
            # Anchor pattern must be full-width for HPGDPullbackMutation's
            # internal margin loss; stable positions are pinned to the
            # anchor input's own natural sign there (they can never move
            # regardless, so this is a no-op "don't care" fill-in, not an
            # assumption about their true value elsewhere in the box).
            full_pattern = _relu_sign_pattern_batched(self.model, anchor_tensor)
        full_pattern[:, self.state_manager.candidate_indices] = restricted_anchor_patterns

        self.gce_mutation.set_anchors(full_pattern)
        inputs = self.gce_mutation.mutate(anchor_tensor, self.model)

        with torch.no_grad():
            output = self.model(inputs)
        outputs = output["output"] if isinstance(output, dict) else output
        violation_mask, counterexamples = self.property_checker.check(inputs=inputs, outputs=outputs, seeds=anchors)

        with torch.no_grad():
            achieved_pattern = _relu_sign_pattern_batched(self.model, inputs)
        admitted = self._observe_state(anchors, inputs, full_pattern, achieved_pattern, violation_mask)
        energies = torch.clamp(admitted.float() * 10.0 + violation_mask.float() * 100.0, min=0.1)

        for ce in counterexamples:
            self.counterexamples.append(ce)
            if self.config.verbose >= 2:
                print(f"🚨 [GCE] Counterexample #{len(self.counterexamples)}: {ce.summary()}")
            if self.config.save_counterexamples:
                self.config.output_dir.mkdir(parents=True, exist_ok=True)
                ce.save(self.config.output_dir / f"ce_{len(self.counterexamples)}.pt")

        child_seeds = FuzzingSeed(
            tensor=inputs,
            original_tensor=anchors.original_tensor,
            original_index=anchors.original_index,
            label=anchors.label,
            energy=energies,
            depth=anchors.depth + 1,
            parent_id=anchors.id,
        )
        self.seed_corpus.add(child_seeds, admitted | violation_mask)

    def _print_progress(self, iteration: int):
        """Print fuzzing progress with incremental counterexample count."""
        elapsed = time.time() - self.start_time
        iter_per_sec = iteration / elapsed if elapsed > 0 else 0
        glc = self.coverage_tracker.get_coverage(strategy="GlobalCov")
        bic = self.coverage_tracker.get_coverage(strategy="BestInputCov")

        # Calculate new counterexamples since last report
        ce_total = len(self.counterexamples)
        ce_new = ce_total - self.last_report_ce_count
        self.last_report_ce_count = ce_total

        samples_per_sec = iter_per_sec * self.batch_size
        print(
            f"📊 Iteration {iteration:6d} | "
            f"GlobalCov: {glc:6.2%} BestInputCov: {bic:6.2%} | "
            f"Seeds: {len(self.seed_corpus):4d} | "
            f"Violations: {ce_total:3d} (+{ce_new}) | "
            f"Speed: {iter_per_sec:5.1f} it/s ({samples_per_sec:.0f} samples/s)"
        )

    def _generate_report(self) -> FuzzingReport:
        """Generate final report."""
        total_time = time.time() - self.start_time

        # Neurons that were never activated across all iterations
        never_activated_neurons: List[Tuple[str, int]] = []
        never_activated_count = 0
        try:
            uncovered = self.coverage_tracker.get_uncovered_neurons()
            never_activated_count = len(uncovered)
            # Deterministic small sample for logs/report
            never_activated_neurons = sorted(list(uncovered))[:20]
        except Exception:
            never_activated_count = 0
            never_activated_neurons = []

        report = FuzzingReport(
            total_iterations=self.iterations,
            total_time=total_time,
            counterexamples=self.counterexamples,
            neuron_coverage=self.coverage_tracker.get_coverage(),
            total_mutations=self.mutation_engine.total_mutations,
            seeds_explored=len(self.seed_corpus),
            num_of_never_activated_neurons=never_activated_count,
            never_activated_neurons=never_activated_neurons,
        )

        # Print summary
        print(f"\n{rule()}")
        print(f"🎉 ACTFuzzer completed in {total_time:.1f}s")
        print(f"   Iterations: {report.total_iterations}")
        print(f"   Counterexamples: {len(report.counterexamples)}")
        bic = self.coverage_tracker.get_coverage(strategy="BestInputCov")
        print(f"   GlobalCov: {report.neuron_coverage:.2%}  BestInputCov: {bic:.2%}")
        print(f"   Seeds explored: {report.seeds_explored}")
        print(f"   Never-activated neurons: {report.num_of_never_activated_neurons}")
        if report.never_activated_neurons:
            sample_str = ", ".join(
                [f"{ln}[{i}]" for (ln, i) in report.never_activated_neurons[:10]]
            )
            print(f"   Never-activated sample: {sample_str}")
        print(f"{rule()}\n")

        if self.config.save_counterexamples and report.counterexamples:
            report.save(self.config.output_dir)

        # Close tracer if enabled
        if self.tracer:
            self.tracer.close()

        return report


def pgd_preattack(wrapped_model, seeds, budget, scale=0.5):
    """Anisotropic-PGD pre-attack (the FALSIFY path of the VNN-COMP runner);
    returns (counterexample_or_None, elapsed_s)."""
    from act.util.device_manager import get_default_device

    device = get_default_device()
    wrapped_model = wrapped_model.to(device)
    seeds = [
        type(s)(tensor=s.tensor.to(device), label=(s.label.to(device) if s.label is not None else None))
        for s in seeds
    ]
    cfg = FuzzingConfig.from_yaml(
        timeout_seconds=float(budget),
        max_iterations=10_000_000,
        mutation_weights={"gradient": 0.0, "pgd": 1.0, "activation": 0.0,
                          "boundary": 0.0, "random": 0.0},
        perturb_mode="adaptive_perdim",
        perturb_scale=scale,
        save_counterexamples=False,
        stop_on_first_violation=True,
        verbose=0,
    )
    report = ACTFuzzer(wrapped_model=wrapped_model, initial_seeds=seeds, config=cfg).fuzz()
    ce = report.counterexamples[0] if report.counterexamples else None
    return ce, report.total_time
