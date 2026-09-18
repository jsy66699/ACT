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
    #   "always" admits every child -- the control for "does admission filter
    #     anything worth filtering", with the energy formula left alone.
    #     Note it keeps COVERAGE's energy formula, so it is "coverage admission
    #     opened to 100%", not "state opened to 100%".
    #   "state_always" is that missing cell: state's energy assignment
    #     (admitted*10) with admission opened to 100%. Since "state" already
    #     admits ~96% of samples, this separates the two things "state" changes
    #     at once -- what gets in, and what it is then worth. Anything it
    #     lets in that "state" rejected lands at the 0.1 floor.
    admission_mode: str = "coverage"
    # How a pre-activation becomes a state coordinate. 2 = sign(z), the ReLU
    # partition and the default everywhere. 3 splits at z = -tau, +tau as well,
    # which on a smooth activation is the only split most neurons can reach --
    # see act/pipeline/fuzzing/state_bins.py for the encoding and the measured
    # reachability. Affects the state code EVERYWHERE it is used: admission,
    # the energy formula's coordinates and HPGD's hinge target.
    state_bins: int = 2
    state_bin_tau: float = 1.0
    # Where the candidate set comes from. "ibp" (default) is interval
    # propagation through the back_end, unchanged. "gradient" uses the box's
    # first-order budget for moving each pre-activation instead -- not sound,
    # but interval propagation has no resolution left on an attention graph
    # (vit_2023: 960/960 unstable, widths of 1e12 against a true 0.10) and the
    # mask only chooses attack targets. See compute_gradient_budget_masks.
    unstable_mask_source: str = "ibp"
    unstable_mask_grad_threshold: float = 1.0
    # select_with_replacement: the paper draws the B lanes with replacement
    # (Algorithm 3 line 10, "high-energy seeds may repeat"), which is the right
    # exploitation policy when every seed belongs to ONE program under test. At
    # B>1 the same draw allocates budget across B INDEPENDENT verification
    # problems, and repetition becomes winner-take-all: measured on
    # tinyimagenet, ~12 counterexample seeds took ~161 of 200 lanes while the
    # 200 per-instance initial seeds shared 0.01% of the sampling mass. Even
    # the very first batch, with all energies still equal, reaches only
    # B(1-1/e) = 63% of the instances. False draws B distinct rows instead
    # (still energy-weighted), so that first batch is a permutation and no
    # single lineage can hold more than one lane. Default True = unchanged.
    select_with_replacement: bool = True
    # Instance-level fairness: round-robin over instances, energy-weighted
    # only within each. select_with_replacement=False was not enough --
    # it draws distinct ROWS, and one instance owns thousands of them, so
    # measured on mnist it left the allocation untouched (Gini 0.77 ->
    # 0.76). With B equal to the instance count every batch becomes a
    # permutation of the instances and starvation is impossible.
    select_per_instance: bool = False
    # scheduling_mode: "energy" (default, original behavior) keeps
    #   SeedCorpus's own energy-weighted select(). "sparse" instead draws
    #   seeds from PatternStateManager.pick_seeds() (density-weighted,
    #   optionally blended with each seed's energy_bonus).
    scheduling_mode: str = "energy"
    # ce_parent_replacement: fixes the energy starvation that makes a single
    # run reach only ~22% of the instances the same arm demonstrably reaches
    # over 30 runs. Energies are admitted*10 + violation*100 (clamped to 0.1)
    # and nothing prunes, so a cracked instance's counterexample lineage
    # compounds until it owns ~95% of SeedCorpus.select()'s mass and the
    # starved instances' 0.1-energy seeds are never drawn again. When on, a CE
    # child whose parent is already in the CE tier REPLACES that parent (one
    # for one per parent) instead of being appended beside it; every seed
    # below the threshold is untouched, so new/uncracked instances keep their
    # slots. Off by default -- the earlier arms' scheduling is unchanged.
    # Scope GlobalCov's "already covered" set per verification instance
    # instead of one union over the whole batch. Off by default: the union is
    # what every earlier arm ran. With a union, instance A firing a neuron
    # raises the admission bar for the other 99 for the rest of the run, and
    # the bar keeps rising as the batch explores -- so the same benchmark gets
    # a harsher rule purely for being run in a bigger batch. At B=1 the two
    # are identical, which is why it only appears once batching exists. State
    # admission is already per instance; this makes the coverage path agree.
    # Reported neuron coverage stays the union either way, so the number
    # remains comparable across arms -- only admission changes.
    coverage_per_instance: bool = False
    # Energy a counterexample seed carries, against an admitted seed's 10 and
    # the 0.1 floor. The shipped 100 makes one CE seed outweigh ten admitted
    # ones, and since nothing prunes, the CE tier ends up owning the draw --
    # measured on safenlp, 100% of the live corpus is counterexamples and
    # 99-100% of the sampling mass. Lowering this is the direct lever on that,
    # complementary to ce_parent_replacement which caps the tier's SIZE
    # instead of its per-seed weight.
    #
    # Keep ce_parent_energy_threshold at or below this value, or replacement
    # silently stops firing.
    ce_energy_bonus: float = 100.0
    # Four explicit energy tiers "plain,admitted,ce,ce_and_admitted", replacing
    # the additive `admitted*10 + violation*ce_energy_bonus` clamped to 0.1.
    # That formula spans {0.1, 10, 100, 110}: a counterexample outweighs a
    # plain seed 1000:1 and an admitted one 10:1, which is what lets the CE
    # tier own the draw. Tiers let the RATIOS be set directly -- "1,5,10,15"
    # keeps the same ordering while making a counterexample worth twice an
    # admitted seed instead of ten times. Empty = the original formula.
    energy_tiers: str = ""
    ce_parent_replacement: bool = False
    # The energy at or above which a parent counts as "already in the CE
    # tier". Only four energies exist -- {0.1, 10, 100, 110} -- so 100 is the
    # whole CE tier (violation with or without a novel pattern) and 110 would
    # narrow it to CE-and-novel only.
    ce_parent_energy_threshold: float = 100.0
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
    # Which spec rows the unstable-neuron mask is derived from.
    #   "row0" (default, what every arm before 2026-08-27 ran): only the first
    #     row's box, applied to all lanes. Measured: a lane's own unstable set
    #     overlaps it by 9-12%, and only 6.6-7.7% of the sign flips that
    #     actually happen land inside it.
    #   "union": every row's box, ORed -- a true superset of each lane's own
    #     set, so the guidance can at least reach the neurons that lane can
    #     flip. Costs a larger candidate set everywhere downstream.
    unstable_mask_scope: str = "row0"
    # Thin each instance's candidate set down to this many state dimensions.
    # 0 = keep all (the shipped behaviour); <1 = a fraction of the instance's
    # own unstable set; >=1 = an absolute count. Only meaningful with
    # unstable_mask_scope="per_instance", which is what gives each instance a
    # set of its own to thin.
    state_dims: float = 0.0
    # How the kept dimensions are chosen.
    #   "random"     -- uniform from the instance's own set. The control: it
    #                   isolates "fewer dimensions" from "these dimensions".
    #   "flip_freq"  -- most balanced sign under random points of the box.
    #   "flip_rare"  -- least balanced that still move at all.
    #   "margin_grad" -- largest |d severity / d pre-activation|: the neurons
    #                   that matter to VIOLATING the property, rather than the
    #                   ones that merely move. The movement-based criteria all
    #                   left distinct unchanged.
    #   "ce_prior"   -- scores harvested from a PREVIOUS run's counterexamples
    #                   (ce_prior_path). An oracle: it uses information from the
    #                   answer, and exists to test whether the information is
    #                   worth anything at all before anyone builds an online
    #                   approximation of it.
    state_dim_select: str = "random"
    ce_prior_path: str = ""
    # Which model group's entry to read out of that file. Instance indices are
    # per group, so reading another group's entry would score the wrong
    # neurons under the right-looking indices.
    ce_prior_key: str = ""
    # Where to write this run's own harvested prior, if anywhere.
    dump_ce_prior_path: str = ""
    # HPGD tuning (the "hpgd" MutationEngine strategy).
    hpgd_flip_count: int = 10
    # When set, HPGD's flip budget is round(hpgd_flip_frac * |candidates|)
    # instead of the fixed hpgd_flip_count. A fixed count is not comparable
    # across networks: 10 flips is 10.8% of safenlp's 93-neuron unstable set
    # but 0.31% of a cifar100 ResNet's 3225, so the same nominal setting asks
    # for displacements two orders of magnitude apart. None = fixed count.
    hpgd_flip_frac: Optional[float] = None
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
    # Weight HPGD's flip choice by per-neuron marginal occupancy, so it aims at
    # under-explored regions instead of drawing k positions uniformly. Combined
    # multiplicatively with local_bias; see PatternStateManager.sparsity_weights.
    hpgd_sparse_targets: bool = True
    # Where HPGD's target pattern comes from.
    #   "random_flip" (default, unchanged): flip k bits of the seed's own
    #     pattern independently. Measured on cifar100_2024: reached ~15% of the
    #     asked flips, exactly ~0%, and the projection ends FARTHER from its
    #     own target than it started (residual/d 1.29-1.71) -- it is chasing a
    #     sign assignment nothing guarantees is satisfiable.
    #   "interp_real": PatternStateManager.propose_targets() names a cell
    #     BETWEEN the seed and one of the SAME instance's real recorded states
    #     -- feasible by construction, unvisited by filter. Measured: reached
    #     exactly ~95-97%. Falls back to random_flip whenever no lane has a
    #     same-instance neighbour yet (e.g. the first iterations).
    hpgd_target_mode: str = "random_flip"
    # Shares of the difference set "interp_real" flips, and how many registry
    # neighbours it draws per lane; every (neighbour, rho) pair is one scored
    # candidate. More candidates = a better pick, at a linear cost.
    hpgd_interp_rhos: str = "0.25,0.5,0.75"
    hpgd_interp_proposals: int = 2
    # Registry entries subsampled per lane when proposing. The registry is
    # scanned per lane, so this is what bounds the cost.
    hpgd_interp_pool: int = 32
    # hpgd_schedule: "off" (default, unchanged) uses hpgd_target_mode as-is for
    #   every lane on every iteration.
    #   "coarse_to_fine" gives each INSTANCE its own two-phase curriculum:
    #     COARSE -- large random flips (hpgd_expand_frac of the candidate set).
    #       Imprecise by design: ~15% of asked flips land. What it buys is
    #       DISPLACEMENT, which pushes that instance's frontier outward and
    #       stocks the registry with the distant real states interpolation
    #       needs as endpoints. Without this phase there is nothing to
    #       interpolate between -- measured, a lane's nearest recorded
    #       neighbour sits ~2-5 sign bits away and its farthest only ~15-33.
    #     FINE -- once the frontier stops growing (hpgd_expand_patience
    #       admissions with no radius gain), that instance switches to
    #       interpolated targets, which are reached exactly ~94-99% of the
    #       time, with rho annealed from hpgd_refine_rho_start down to
    #       hpgd_refine_rho_end so precision rises as the phase goes on.
    #   The phase is per instance and reversible: an instance whose radius
    #   starts growing again drops back to coarse by itself.
    hpgd_schedule: str = "off"
    hpgd_expand_frac: float = 0.05
    hpgd_expand_patience: int = 20
    hpgd_refine_rho_start: float = 0.75
    hpgd_refine_rho_end: float = 0.25
    # A fine-phase lane needs a neighbour at least this far away; closer ones
    # leave nothing to interpolate.
    hpgd_interp_min_d: int = 4
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


def _load_ce_prior(path: str, key: str):
    """The harvested prior for one model group, or {} when there is none."""
    import os
    if not path or not os.path.exists(path):
        return {}
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    if key and key in blob:
        return blob[key]
    # One group in the file, or a key mismatch: only unambiguous when there is
    # exactly one, since instance indices are per group and mixing them would
    # silently score the wrong neurons.
    return next(iter(blob.values())) if len(blob) == 1 else {}


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
            # One coverage row per instance instead of a single batch union.
            # batch_size is the spec-row count, which is exactly the number of
            # verification instances in this group.
            per_instance=self.batch_size if self.config.coverage_per_instance else 0,
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

        # Only wired when "hpgd" can actually be drawn -- local_bias_batch
        # hashes every seed in the batch, which is wasted work at weight 0.
        self._hpgd_strategy = (
            self.mutation_engine.strategies.get("hpgd")
            if float(self.config.mutation_weights.get("hpgd", 0.0)) > 0 else None
        )

        # Split by how the target was chosen, because pooling them is
        # misleading: "landed % of asked" is asked-WEIGHTED, and a coarse lane
        # asks for ~100 flips against a named lane's ~10, so 3 landed flips out
        # of 10 named requests disappear behind 98 unlanded coarse ones.
        #
        # Also counted twice over different scopes. The full pattern is every
        # ReLU in the network; `_c` is restricted to the unstable candidate
        # subspace that HPGD aims in and admission scores in (medium: 951 of
        # 55,460). Measured collateral says most flips that actually happen
        # land OUTSIDE that subspace, which would mean the strategy steers a
        # small corner of the state space it thinks it controls -- `outside`
        # is the count that settles it.
        def _diag_row():
            return {"lanes": 0, "asked": 0, "landed": 0, "collateral": 0,
                    "asked_c": 0, "landed_c": 0, "collateral_c": 0,
                    "reached": 0, "reached_c": 0, "outside": 0,
                    "target_seen": 0, "achieved_seen": 0, "moved": 0}

        # What the novelty predicate actually decided, before the gate folds it
        # together with the violation bypass. The reported rejection rate
        # (1 - (rows-B)/iterations, or corpus_drops["gate"]) cannot answer this:
        # `interesting_mask = violation | admitted` lets every counterexample in
        # regardless of novelty, so a benchmark whose children are mostly
        # counterexamples shows a low "rejection" no matter how strict the
        # predicate is -- tinyimagenet's statebase reads 7.1% only because 88%
        # of its children were violations. These four count the predicate's own
        # verdict on every sample it saw.
        # Bucketed by the strategy that produced the batch. MutationEngine
        # samples ONE strategy per iteration, so a change to any single
        # strategy's targeting shows up diluted by that strategy's share in
        # the aggregate: hpgd at 33.3% of the portfolio moves the overall rate
        # by a third of its own change, and reversing that division needs an
        # assumption about the other strategies that does not hold (the arms'
        # corpora differ, so pgd/boundary/random see different seeds).
        self._novelty = {"observed": 0, "admitted": 0, "ce": 0, "ce_and_novel": 0}
        self._novelty_by_strategy: dict[str, dict[str, int]] = {}
        self._hpgd_diag = (
            {"calls": 0, "named": _diag_row(), "random": _diag_row()}
            if self._hpgd_strategy is not None else None
        )

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
        # Built in _setup_state once the neuron count is known; None means the
        # two-bin sign(z) code, which is what every non-state arm uses.
        self.binning = None
        needs_state_manager = (
            self.config.admission_mode in ("state", "state_always")
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

        # Every probe forward below goes through the WRAPPED model, which
        # evaluates its output spec as part of forward(). A row-indexed spec
        # (TOP1_ROBUST's y_true, one row per synthesized instance) rejects any
        # batch whose lane count differs from its row count, so these probes
        # must present all self.batch_size lanes -- one seed is not enough.
        # Probing with a single seed raised
        #   "TOP1_ROBUST: y_true carries B spec rows but the batch has 1 lanes"
        # which crashed the fuzzer outright here and, three lines down, was
        # swallowed by a bare `except Exception` into a silently dropped
        # unstable mask. safenlp's UNSAFE_LINEAR spec is not row-indexed the
        # same way, which is why every earlier state-admission campaign missed
        # both.
        sample = torch.cat([s.tensor.to(self.device) for s in initial_seeds], dim=0)
        with torch.no_grad():
            total_neurons = int(_relu_preactivations_batched(self.model, sample).shape[1])

        from act.pipeline.fuzzing.state_bins import StateBinning
        self.binning = StateBinning.build(
            total_neurons, bins=self.config.state_bins,
            tau=self.config.state_bin_tau, device=self.device,
        )
        # Everything downstream counts COORDINATES, not neurons: the mask, the
        # BK-tree width, the candidate set and HPGD's targets all live in the
        # coordinate space, and under bins=3 that is twice as wide.
        total_coords = self.binning.num_coords
        if self.config.state_bins != 2:
            print(f"   [state] {self.config.state_bins}-bin state at tau="
                  f"{self.config.state_bin_tau}: {total_neurons} neurons -> "
                  f"{total_coords} coordinates")

        unstable_mask = None
        mask_reason = "no_input_spec"
        if self.input_spec is not None:
            try:
                lb, ub = self.input_spec.materialize_box_seed()
                scope = self.config.unstable_mask_scope
                per_instance_mask = None
                if self.config.unstable_mask_source == "gradient":
                    from act.pipeline.fuzzing.state_manager import (
                        compute_gradient_budget_masks,
                    )
                    per_instance_mask, mask_reason = compute_gradient_budget_masks(
                        self.model, lb.to(self.device), ub.to(self.device),
                        binning=self.binning,
                        threshold=self.config.unstable_mask_grad_threshold)
                    if per_instance_mask is not None:
                        # The criterion is per lane by construction, so "row0"
                        # would be throwing away rows that cost nothing extra.
                        # union keeps the scope flag meaningful for callers
                        # that ask for it; anything else takes lane 0 as before.
                        unstable_mask = (per_instance_mask.any(dim=0)
                                         if scope in ("union", "per_instance")
                                         else per_instance_mask[0])
                        if scope != "per_instance":
                            per_instance_mask = None
                    else:
                        unstable_mask = None
                elif scope == "per_instance":
                    from act.pipeline.fuzzing.state_manager import compute_per_instance_masks
                    per_instance_mask, mask_reason = compute_per_instance_masks(
                        self.model, lb.to(self.device), ub.to(self.device),
                        binning=self.binning)
                    unstable_mask = (per_instance_mask.any(dim=0)
                                     if per_instance_mask is not None else None)
                else:
                    unstable_mask, mask_reason = compute_unstable_mask(
                        self.model, lb.to(self.device), ub.to(self.device), scope=scope,
                        binning=self.binning,
                    )
            except Exception as exc:
                unstable_mask = None
                mask_reason = f"materialize_box_seed_failed: {type(exc).__name__}: {exc}"

        if unstable_mask is None:
            print(
                f"   [state] unstable mask UNAVAILABLE ({mask_reason}) -- "
                f"falling back to all {total_coords} coordinates as flip candidates"
            )
        else:
            n_unstable = int(unstable_mask.sum())
            print(
                f"   [state] unstable coordinates: {n_unstable}/{total_coords} "
                f"({n_unstable / max(total_coords, 1):.1%}) -- "
                f"{total_coords - n_unstable} pinned on this box"
            )

        self._per_instance_mask = per_instance_mask
        self._total_neurons = int(total_neurons)
        self.state_manager = PatternStateManager(
            unstable_mask=unstable_mask,
            total_neurons=total_coords,
            diversity_threshold=self.config.state_diversity_threshold,
            bloom_bits=self.config.state_bloom_bits,
            bloom_hashes=self.config.state_bloom_hashes,
            local_bias_high=self.config.state_local_bias_high,
            local_bias_low=self.config.state_local_bias_low,
            device=self.device,
        )

        if self._per_instance_mask is not None and self.config.state_dims > 0:
            from act.pipeline.fuzzing.state_manager import (
                flip_balance_scores, select_state_dims,
            )
            _scores = None
            if self.config.state_dim_select == "ce_prior":
                _prior = _load_ce_prior(self.config.ce_prior_path,
                                        self.config.ce_prior_key)
                if _prior:
                    n_full = self._per_instance_mask.shape[1]
                    rows = []
                    for i in range(self._per_instance_mask.shape[0]):
                        e = _prior.get(i)
                        rows.append(e["score"] if e is not None else torch.zeros(n_full))
                    _scores = torch.stack(rows).to(self.device)
                    _scores = _scores[:, unstable_mask.nonzero(as_tuple=True)[0]]
                    print(f"   [state] ce_prior covers {len(_prior)} instance(s); "
                          f"the rest fall back to a random pick")
                else:
                    print("   [state] ce_prior file empty/missing for this group; "
                          "falling back to a random pick")
            if self.config.state_dim_select == "margin_grad":
                from act.pipeline.fuzzing.state_manager import margin_gradient_scores
                _scores = margin_gradient_scores(
                    self.model, self.output_spec,
                    lb.to(self.device), ub.to(self.device))
                if _scores is not None:
                    _scores = _scores[:, unstable_mask.nonzero(as_tuple=True)[0]]
                else:
                    print("   [state] margin-gradient scores unavailable; "
                          "falling back to a random pick")
            if self.config.state_dim_select in ("flip_freq", "flip_rare"):
                _scores = flip_balance_scores(self.model, lb.to(self.device), ub.to(self.device))
                if _scores is not None:
                    _scores = _scores[:, unstable_mask.nonzero(as_tuple=True)[0]]
            _sub = self._per_instance_mask[:, unstable_mask.nonzero(as_tuple=True)[0]]
            _sub = select_state_dims(_sub, self.config.state_dim_select,
                                     self.config.state_dims, _scores)
            print(f"   [state] state dims per instance: "
                  f"{float(_sub.sum(dim=1).float().mean()):.0f} "
                  f"(select={self.config.state_dim_select})")
            self._per_instance_submask_override = _sub

        if self._per_instance_mask is not None:
            # Same axis the manager restricts to, so a [B, C] lookup lines up
            # with restricted patterns and with HPGD's candidate list.
            override = getattr(self, "_per_instance_submask_override", None)
            self.state_manager.instance_submask = (
                override.contiguous() if override is not None
                else self._per_instance_mask[
                    :, self.state_manager.candidate_indices
                ].contiguous()
            )

        hpgd = self.mutation_engine.strategies.get("hpgd")
        if hpgd is not None:
            hpgd.candidate_indices = self.state_manager.candidate_indices
            # The candidate set indexes coordinates, so the strategy must read
            # the same binning or the two disagree about what index j means.
            hpgd.binning = self.binning if self.config.state_bins != 2 else None
            hpgd.flip_count = self.config.hpgd_flip_count
            hpgd.flip_frac = self.config.hpgd_flip_frac
            hpgd.num_steps = self.config.hpgd_num_steps
            hpgd.margin = self.config.hpgd_margin
            hpgd.loss_scope = self.config.hpgd_loss_scope
            hpgd.hold_still_weight = self.config.hpgd_hold_still_weight

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

        # Exact-match admission is the default (diversity_threshold 1), and it
        # is the case a fingerprint can answer in bulk. Anything larger is a
        # Hamming-ball query, which fingerprints cannot express, so it stays on
        # the per-sample BK-tree path below.
        if max(0, self.state_manager.diversity_threshold - 1) == 0:
            admitted = self.state_manager.observe_batch(
                seed_tensors=child_inputs,
                patterns_full=achieved_pattern,
                labels=parent_seeds.label,
                original_tensors=parent_seeds.original_tensor,
                original_indices=parent_seeds.original_index,
                is_ce_mask=violation_mask,
            )
            if admitted.any():
                restricted_natural = self.state_manager.restrict(natural_pattern)
                restricted_achieved = self.state_manager.restrict(achieved_pattern)
                for b in admitted.nonzero(as_tuple=True)[0].tolist():
                    flipped = (restricted_natural[b] != restricted_achieved[b]).nonzero(as_tuple=True)[0]
                    self.state_manager.update_local_bias(child_inputs[b : b + 1], flipped)
            return admitted

        admitted = torch.zeros(B, dtype=torch.bool, device=child_inputs.device)
        # One transfer instead of B per-element .item() syncs in the loop.
        is_ce_list = violation_mask.tolist()
        for b in range(B):
            is_ce = bool(is_ce_list[b])
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

        # Warm the coverage tracker on the initial corpus before the first
        # mutation. Without this, iteration 0 sees masks that have never been
        # built, which get_uncovered_neurons() reports identically to full
        # saturation -- so any coverage-steered strategy is blind on its first
        # call. Counts toward GlobalCov (the seeds are inputs the fuzzer has
        # genuinely executed), so coverage here is not comparable to runs from
        # before this warmup existed.
        with torch.no_grad():
            _seed_batch = self.seed_corpus.select(batch_size)
            _warm_out = self.model(_seed_batch.tensor.to(self.device))
            del _warm_out
            self.coverage_tracker.update(
                _seed_batch.tensor.to(self.device),
                self.mutation_engine.get_activation_map(),
                rows=(_seed_batch.original_index
                      if self.config.coverage_per_instance else None),
            )
            _other = "BestInputCov" if self.config.coverage_strategy == "GlobalCov" else "GlobalCov"
            self.coverage_tracker.update(
                _seed_batch.tensor.to(self.device),
                self.mutation_engine.get_activation_map(),
                strategy=_other,
            )

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
            seeds = self.seed_corpus.select(
                batch_size, replace=self.config.select_with_replacement,
                per_instance=self.config.select_per_instance)

        # PatternStateManager admission needs the PRE-mutation pattern to
        # later diff against the achieved one (which candidate neurons this
        # specific mutation actually flipped, for local_bias).
        natural_pattern = None
        if (self.config.admission_mode in ("state", "state_always")
                and self.state_manager is not None):
            with torch.no_grad():
                natural_pattern = _relu_sign_pattern_batched(self.model, seeds.tensor.to(self.device),
                                                     binning=self.binning)

        # Close HPGD's feedback loop. update_local_bias() has always recorded,
        # for every admitted sample, which unstable-subspace positions actually
        # flipped to produce it -- but nothing ever read that back, so
        # HPGDMutation.flip_weights stayed None and it picked its K target
        # neurons with a uniform torch.randperm every call. The manager was
        # maintaining a table on every iteration that no one consulted, and the
        # strategy was random-walking while admission scored novelty; the two
        # never talked. Feeding the recorded bias in is what makes HPGD's
        # target selection exploit what proved flippable from THIS seed.
        if self._hpgd_strategy is not None and self.state_manager is not None:
            w = self.state_manager.local_bias_batch(seeds.tensor.to(self.device))
            if self.config.hpgd_sparse_targets and natural_pattern is not None:
                # Multiply the two signals rather than pick one: local_bias says
                # which neurons proved FLIPPABLE from this seed, sparsity says
                # which flips LEAD SOMEWHERE under-explored. Either alone is
                # half the question -- a flippable neuron that returns to a
                # crowded region wastes the projection, and a neuron pointing
                # somewhere novel that will not move wastes it too. An unseen
                # seed has uniform local_bias, so sparsity then decides alone.
                w = w * self.state_manager.sparsity_weights(
                    self.state_manager.restrict(natural_pattern)
                )
            if self.state_manager.instance_submask is not None:
                # Zero out the candidates this lane's own box holds fixed.
                # Without this, a union candidate axis leaves ~5% of the axis
                # flippable for a given lane, so a uniform draw of k targets
                # asks for flips that cannot happen -- the same failure row0
                # had, arrived at from the other direction.
                own = self.state_manager.instance_submask.to(w.device)[
                    seeds.original_index.to(self.state_manager.instance_submask.device)
                ].to(w.device)
                w = w * own.to(w.dtype)
            self._hpgd_strategy.flip_weights = w

            # Name the target explicitly instead of letting HPGD flip k bits
            # blind -- but LAZILY: installed as a callback HPGD invokes from
            # inside its own mutate(), so the registry scan is paid only on the
            # iterations HPGD is actually dispatched on (it holds a third of
            # the portfolio, so eager computation wasted two thirds of it).
            if (self.config.hpgd_schedule == "coarse_to_fine"
                    or self.config.hpgd_target_mode == "interp_real"):
                self._hpgd_strategy.target_proposer = (
                    lambda nat, _s=seeds: self._propose_hpgd_targets(_s, nat)
                )
            else:
                self._hpgd_strategy.target_proposer = None

        # 2. mutate — takes FuzzingSeed, returns Tensor[B, ...]
        inputs = self.mutation_engine.mutate(seeds)

        # Diagnostic: is HPGD spending its 10 gradient steps aiming at states it
        # has already visited? PatternSearchPGD re-flips until the TARGET is
        # unseen (pattern_search_pgd.py:392); HPGDMutation only ever filters the
        # ACHIEVED pattern, at admission, after the cost is sunk. Measure the
        # gap before deciding whether porting that retry is worth it.
        if (self._hpgd_diag is not None
                and self.mutation_engine.last_strategy == "hpgd"
                and self.state_manager is not None):
            hp = self._hpgd_strategy
            if hp is not None and hp.last_target_pattern is not None:
                rows = seeds.original_index.to(self.device)
                # .cpu() is load-bearing: seen_mask builds its result with a
                # bare torch.tensor, so under a torch.device context it lands on
                # the accelerator, while reached/moved below are explicitly CPU.
                # The loop indexes both groups, so they must agree.
                tgt_seen = self.state_manager.seen_mask(hp.last_target_pattern, rows).cpu()
                ach_seen = self.state_manager.seen_mask(hp.last_achieved_pattern, rows).cpu()
                nat, tgt, ach = (hp.last_natural_pattern, hp.last_target_pattern,
                                 hp.last_achieved_pattern)
                reached = (tgt == ach).all(dim=1).cpu()
                moved = (nat != ach).any(dim=1).cpu()
                # The question full-pattern equality is too strict to answer:
                # of the neurons this call ASKED to flip, how many actually
                # flipped? Everything else is collateral.
                asked = (tgt != nat)
                landed = asked & (ach == tgt)
                collateral = (~asked) & (ach != nat)

                cand = self.state_manager.candidate_indices.to(asked.device)
                asked_c = asked[:, cand]
                landed_c = landed[:, cand]
                collateral_c = collateral[:, cand]
                reached_c = (tgt[:, cand] == ach[:, cand]).all(dim=1).cpu()
                # Flips that happened where HPGD cannot aim and admission does
                # not score -- outside the unstable candidate subspace.
                outside = collateral.sum(dim=1) - collateral_c.sum(dim=1)

                use = hp.target_override_mask
                named = (use.cpu() if use is not None and use.shape[0] == asked.shape[0]
                         else torch.zeros(asked.shape[0], dtype=torch.bool, device="cpu"))
                d = self._hpgd_diag
                d["calls"] += 1
                for key, sel in (("named", named), ("random", ~named)):
                    if not bool(sel.any()):
                        continue
                    s_dev = sel.to(asked.device)
                    row = d[key]
                    row["lanes"] += int(sel.sum())
                    row["asked"] += int(asked[s_dev].sum())
                    row["landed"] += int(landed[s_dev].sum())
                    row["collateral"] += int(collateral[s_dev].sum())
                    row["asked_c"] += int(asked_c[s_dev].sum())
                    row["landed_c"] += int(landed_c[s_dev].sum())
                    row["collateral_c"] += int(collateral_c[s_dev].sum())
                    row["outside"] += int(outside[s_dev].sum())
                    row["reached"] += int(reached[sel].sum())
                    row["reached_c"] += int(reached_c[sel].sum())
                    row["target_seen"] += int(tgt_seen[sel].sum())
                    row["achieved_seen"] += int(ach_seen[sel].sum())
                    row["moved"] += int(moved[sel].sum())

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
            inputs, activations,
            rows=seeds.original_index if self.config.coverage_per_instance else None,
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
        if self.config.admission_mode == "always":
            # The control that asks whether admission filters anything worth
            # filtering: every child enters the corpus. Energies keep the
            # coverage formula, so the ONLY thing this changes is who gets in,
            # not how heavily they are then weighted.
            interesting_mask = torch.ones(inputs.shape[0], dtype=torch.bool,
                                          device=cov_interesting.device)
            energies = cov_interesting.float() * 10.0 + violation_mask.float() * self.config.ce_energy_bonus
        elif (self.config.admission_mode in ("state", "state_always")
                and self.state_manager is not None):
            with torch.no_grad():
                achieved_pattern = _relu_sign_pattern_batched(self.model, inputs, binning=self.binning)
            admitted = self._observe_state(seeds, inputs, natural_pattern, achieved_pattern, violation_mask)
            _v = violation_mask.to(admitted.device)
            _counts = {
                "observed": int(admitted.numel()),
                "admitted": int(admitted.sum()),
                "ce": int(_v.sum()),
                "ce_and_novel": int((_v & admitted).sum()),
            }
            for _k, _n in _counts.items():
                self._novelty[_k] += _n
            _s = getattr(self.mutation_engine, "last_strategy", None) or "unknown"
            _b = self._novelty_by_strategy.setdefault(
                _s, {"observed": 0, "admitted": 0, "ce": 0, "ce_and_novel": 0})
            for _k, _n in _counts.items():
                _b[_k] += _n
            # "state_always" keeps this same energy assignment and only opens
            # the gate: what state would have rejected still enters, at the
            # 0.1 floor, so the arm isolates energy from admission.
            interesting_mask = (
                torch.ones_like(admitted) if self.config.admission_mode == "state_always"
                else violation_mask | admitted
            )
            energies = admitted.float() * 10.0 + violation_mask.float() * self.config.ce_energy_bonus
        else:
            interesting_mask = violation_mask | cov_interesting
            energies = cov_interesting.float() * 10.0 + violation_mask.float() * self.config.ce_energy_bonus
        if self.config.energy_tiers:
            t = [float(v) for v in self.config.energy_tiers.split(",")]
            assert len(t) == 4, "energy_tiers wants plain,admitted,ce,ce_and_admitted"
            # Recover the two flags from the additive energies computed above:
            # subtracting the CE term leaves exactly the admitted bonus.
            is_ce = violation_mask.to(energies.device)
            is_adm = (energies - is_ce.float() * self.config.ce_energy_bonus) >= 10.0
            energies = torch.where(
                is_ce & is_adm, torch.full_like(energies, t[3]),
                torch.where(is_ce, torch.full_like(energies, t[2]),
                torch.where(is_adm, torch.full_like(energies, t[1]),
                            torch.full_like(energies, t[0]))))
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
        # The replacement kwargs are passed only when the feature is on, so
        # the call shape stays exactly what it was for anything that swaps
        # add() out (compare_batch_aniso_overapprox's --freeze-seed-corpus).
        self.seed_corpus.add(child_seeds, interesting_mask, **self._ce_add_kwargs(violation_mask))

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
            full_pattern = _relu_sign_pattern_batched(self.model, anchor_tensor, binning=self.binning)
        full_pattern[:, self.state_manager.candidate_indices] = restricted_anchor_patterns

        self.gce_mutation.set_anchors(full_pattern)
        inputs = self.gce_mutation.mutate(anchor_tensor, self.model)

        with torch.no_grad():
            output = self.model(inputs)
        outputs = output["output"] if isinstance(output, dict) else output
        violation_mask, counterexamples = self.property_checker.check(inputs=inputs, outputs=outputs, seeds=anchors)

        with torch.no_grad():
            achieved_pattern = _relu_sign_pattern_batched(self.model, inputs, binning=self.binning)
        admitted = self._observe_state(anchors, inputs, full_pattern, achieved_pattern, violation_mask)
        energies = torch.clamp(admitted.float() * 10.0 + violation_mask.float() * self.config.ce_energy_bonus, min=0.1)

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
        self.seed_corpus.add(child_seeds, admitted | violation_mask,
                             **self._ce_add_kwargs(violation_mask))

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
        # Coarse-to-fine only: how the batch split between the two roles, and
        # how far the frontier has actually been pushed. Without this the
        # schedule is unfalsifiable -- "it ran" says nothing about whether any
        # instance ever left the coarse phase.
        phase = getattr(self, "_hpgd_phase", None)
        if phase is not None and self.state_manager is not None:
            fine, total = phase
            radii = [self.state_manager.frontier(i)[0]
                     for i in self.state_manager._inst_radius]
            if radii:
                print(f"   [hpgd] fine {fine}/{total} lanes | frontier radius "
                      f"mean {sum(radii)/len(radii):.1f} max {max(radii)} "
                      f"over {len(radii)} instances")

    def _propose_hpgd_targets(self, seeds: FuzzingSeed, natural_pattern: torch.Tensor):
        """HPGD's per-lane (target, mask, flip_counts) for this call.

        Three configurations land here:
          hpgd_schedule "off" + target_mode "random_flip" -- clears both seams,
            so HPGD builds its own target exactly as it always did.
          hpgd_schedule "off" + target_mode "interp_real" -- every lane that
            has a usable neighbour gets an interpolated target; the rest fall
            back to random flips via the per-lane mask (leaving them unmasked
            would hand them a target identical to their own pattern, asking for
            zero flips and wasting the whole projection step on that lane).
          hpgd_schedule "coarse_to_fine" -- the mask is additionally gated on
            each instance's own phase, and coarse lanes get the large expand
            budget instead of hpgd_flip_count.
        """
        sm = self.state_manager
        scheduled = self.config.hpgd_schedule == "coarse_to_fine"
        if sm is None or (not scheduled
                          and self.config.hpgd_target_mode != "interp_real"):
            return None, None, None

        B = natural_pattern.shape[0]
        device = natural_pattern.device
        candidates = sm.candidate_indices
        want_fine = (sm.refine_mask(seeds.original_index, self.config.hpgd_expand_patience)
                     if scheduled else torch.ones(B, dtype=torch.bool))
        want_fine = want_fine.to(device)

        rho_per_lane = None
        if scheduled:
            # Anneal within the fine phase: the longer an instance has gone
            # without growing its frontier, the smaller a step it takes, so
            # precision rises as expansion dries up. Capped at 2x patience so
            # rho settles at its end value instead of collapsing to nothing.
            patience = max(1, self.config.hpgd_expand_patience)
            stalls = torch.tensor(
                [sm.frontier(int(i))[1] for i in seeds.original_index.tolist()],
                dtype=torch.float32, device=device,
            )
            t = ((stalls - patience) / patience).clamp(0.0, 1.0)
            rho_per_lane = (self.config.hpgd_refine_rho_start
                            + t * (self.config.hpgd_refine_rho_end
                                   - self.config.hpgd_refine_rho_start))

        rhos = tuple(float(r) for r in self.config.hpgd_interp_rhos.split(",") if r.strip())
        targets, proposed = sm.propose_targets(
            sm.restrict(natural_pattern),
            seeds.original_index,
            rhos=rhos,
            pool=self.config.hpgd_interp_pool,
            proposals=self.config.hpgd_interp_proposals,
            min_d=self.config.hpgd_interp_min_d,
            rho_per_lane=rho_per_lane,
        )
        use = want_fine & proposed.to(device)

        # propose_targets works in the restricted subspace; HPGD wants a
        # full-length pattern, so scatter it back onto the natural one
        # (untouched outside the candidate dims).
        full = natural_pattern.clone()
        full[:, candidates] = targets.to(full.dtype).to(full.device)

        counts = None
        if scheduled:
            expand_k = max(1, int(round(self.config.hpgd_expand_frac * candidates.numel())))
            counts = torch.where(
                use, torch.full((B,), self.config.hpgd_flip_count, device=device),
                torch.full((B,), expand_k, device=device),
            )

        # Recorded here rather than at call-installation time, so the reported
        # split is what HPGD actually did, not what it would have done on an
        # iteration that dispatched to another strategy.
        self._hpgd_phase = (int(use.sum()), B)
        return full, use, counts

    def _ce_add_kwargs(self, violation_mask: torch.Tensor) -> dict:
        """SeedCorpus.add() kwargs. ce_mask goes in on every call so the corpus
        can flag which rows are counterexamples -- ce_count/ce_mass need that
        whether or not replacement is on, and under a ce_energy below the
        admitted bonus no energy threshold can recover it. Retirement stays
        gated on the feature flag. (The one thing that swaps add() out,
        compare_batch_aniso_overapprox's --freeze-seed-corpus, takes **kwargs.)"""
        return {
            "ce_mask": violation_mask,
            "replace_ce_parents": self.config.ce_parent_replacement,
        }

    def dump_ce_patterns(self, path: str, key: str) -> int:
        """Raw per-instance sign patterns, split by counterexample or not.

        dump_ce_prior only keeps a per-neuron summary, which cannot answer
        whether an instance's counterexamples sit in ONE linear region or many:
        that is a question about the joint pattern, not about marginals."""
        if self.state_manager is None or not path:
            return 0
        import os
        by = {}
        for node in self.state_manager._registry:
            inst = int(node.payload["original_index"].reshape(-1)[0])
            d = by.setdefault(inst, {"ce": [], "non": []})
            d["ce" if node.payload.get("is_ce") else "non"].append(
                node.point.detach().to(torch.int8).cpu())
        out = {}
        for inst, d in by.items():
            if not d["ce"]:
                continue
            out[inst] = {
                "ce": torch.stack(d["ce"]),
                # Cap the non-CE side: it is only a reference population and a
                # cracked instance can record thousands.
                "non": torch.stack(d["non"][:2000]) if d["non"] else None,
            }
        blob = {}
        if os.path.exists(path):
            try:
                blob = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                blob = {}
        g = blob.setdefault(key, {})
        for inst, e in out.items():
            if inst in g:
                g[inst] = {"ce": torch.cat([g[inst]["ce"], e["ce"]]),
                           "non": e["non"] if g[inst]["non"] is None else g[inst]["non"]}
            else:
                g[inst] = e
        blob["__candidates__" + key] = self.state_manager.candidate_indices.cpu()
        torch.save(blob, path)
        return len(out)

    def dump_ce_prior(self, path: str, key: str) -> int:
        """Append this run's per-instance CE suspiciousness to `path`, keyed by
        model group. Returns how many instances had both populations."""
        if self.state_manager is None or not path:
            return 0
        from act.pipeline.fuzzing.state_manager import ce_prior_from_registry
        import os

        prior = ce_prior_from_registry(self.state_manager, self._total_neurons)
        blob = {}
        if os.path.exists(path):
            try:
                blob = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                blob = {}
        group = blob.setdefault(key, {})
        for inst, entry in prior.items():
            # Merge across runs by summing the evidence: more counterexamples
            # from more runs should sharpen the score, not overwrite it.
            prev = group.get(inst)
            if prev is None:
                group[inst] = entry
            else:
                w0, w1 = prev["n_ce"], entry["n_ce"]
                group[inst] = {
                    "score": (prev["score"] * w0 + entry["score"] * w1) / max(1, w0 + w1),
                    "ce_sign": entry["ce_sign"] if w1 >= w0 else prev["ce_sign"],
                    "n_ce": w0 + w1,
                    "n_non": prev["n_non"] + entry["n_non"],
                }
        torch.save(blob, path)
        return len(prior)

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
            # rows, not len(): retirement shrinks the live pool, and an arm
            # that retires must not read as having explored fewer seeds.
            # Identical to len() whenever ce_parent_replacement is off.
            seeds_explored=self.seed_corpus.rows,
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

        # A saturated benchmark turns hpgd_cov into plain Gaussian noise, which
        # is invisible in every other number here -- say so out loud rather than
        # letting an experiment arm quietly measure something else.
        _cov = self.mutation_engine.strategies.get("hpgd_cov")
        if _cov is not None and sum(_cov.fallback_counts.values()) > 0:
            fc = _cov.fallback_counts
            total = sum(fc.values())
            print(f"   HPGD-Cov calls: {total} "
                  f"(ran {fc['ran']}, fully-covered {fc['fully_covered']}, "
                  f"pre-warmup {fc['not_yet_observed']}, no-tracker {fc['no_tracker']})")
            if fc["ran"] == 0:
                print("   ⚠️  HPGD-Cov never steered: it ran as Gaussian noise "
                      "for the whole run (coverage saturated?).")

        d = self._hpgd_diag
        if d is not None and (d["named"]["lanes"] + d["random"]["lanes"]) > 0:
            print(f"   HPGD targets: {d['calls']} calls | "
                  f"named {d['named']['lanes']} lanes / "
                  f"random {d['random']['lanes']} lanes")
            for key in ("named", "random"):
                r = d[key]
                L = r["lanes"]
                if L == 0:
                    continue
                a, ac = max(r["asked"], 1), max(r["asked_c"], 1)
                print(f"   [{key}] asked {r['asked'] / L:6.2f}/lane "
                      f"({r['asked_c'] / L:6.2f} in candidates) | "
                      f"landed {r['landed'] / a:6.1%} "
                      f"({r['landed_c'] / ac:6.1%} in candidates)")
                print(f"   [{key}] reached {r['reached'] / L:6.1%} full "
                      f"({r['reached_c'] / L:6.1%} on candidates) | "
                      f"target seen {r['target_seen'] / L:5.1%} | "
                      f"achieved seen {r['achieved_seen'] / L:5.1%}")
                print(f"   [{key}] collateral {r['collateral'] / L:6.2f}/lane, "
                      f"of which {r['outside'] / L:6.2f} OUTSIDE the candidate "
                      f"subspace | moved {r['moved'] / L:5.1%}")
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
