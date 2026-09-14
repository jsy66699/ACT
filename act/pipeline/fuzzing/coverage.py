"""
Coverage tracking for ACTFuzzer.

Tracks neuron coverage (method-level metrics) during fuzzing to guide exploration.

## How Coverage Is Collected

1. **Activation Capture**: MutationEngine registers forward hooks on computational layers
   (ReLU, Linear, Conv2d). During inference, these hooks capture layer activations into
   an activation_map dict keyed by layer name.

2. **Coverage Update**: After inference, ACTFuzzer calls CoverageTracker.update() with the
   mutated input and the activation_map. The tracker delegates to the active CoverageStrategy.

3. **Neuron Firing**: A neuron is considered "fired" (covered) if |activation| > threshold.
   For multi-dimensional activations (e.g., Conv2d [N,C,H,W]), spatial dims are max-pooled
   to produce per-channel coverage: one neuron per channel.

4. **Coverage Delta**: update() returns (global_delta, interesting_mask) — the coverage
   improvement and a per-sample BoolTensor marking which samples are interesting.
   ACTFuzzer uses these for energy computation and corpus scheduling.

## Coverage Strategies

1. **BestInputCov (BIC)**: Per-input coverage tracking. Each input gets its own coverage
   score (fraction of neurons fired). get_coverage() returns the best (max) per-input
   coverage seen so far. Does NOT maintain a global union — only individual input scores.

2. **GlobalCov (GLC)**: Global union coverage. A neuron is covered once it fires in ANY
   input across ALL iterations (monotonic). Coverage state is stored as per-layer
   BoolTensors on the activation device for O(1) vectorized lookup.

## Statistics

get_stats() returns strategy-specific metrics:
- BestInputCov: coverage, inputs_seen, last/best/avg input coverage, total neurons, layers
- GlobalCov: coverage, covered/total neurons, last newly covered count, layers

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set, Tuple
import torch
import torch.nn as nn


NeuronId = Tuple[str, int]


class CoverageStrategy(ABC):
    """
    Coverage strategy interface.

    A strategy owns its own coverage state (covered set, totals, stats) and can be
    plugged into a tracker/engine (similar to `MutationStrategy` + `MutationEngine`).
    """

    def __init__(self, model: nn.Module, threshold: float = 0.1):
        self.model = model
        self.threshold = threshold

    @abstractmethod
    def update(
        self, input_tensor: torch.Tensor, activations: Dict[str, torch.Tensor]
    ) -> Tuple[float, torch.Tensor]:
        """
        Update coverage with batch activations.
        
        Batch-native: processes all N samples in one vectorized call.
        
        Returns:
            (global_delta, interesting_mask) where:
            - global_delta: overall coverage improvement (float, 0..1)
            - interesting_mask: BoolTensor of shape (N,), True for samples that
              contributed to coverage improvement (should be added to corpus)
        """
        raise NotImplementedError

    @abstractmethod
    def get_coverage(self) -> float:
        """Return coverage in [0, 1]."""
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return strategy-specific coverage stats (JSON friendly)."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset internal coverage state."""
        raise NotImplementedError

    # Optional capabilities (implemented only by some strategies)
    def get_uncovered_neurons(self) -> Set[NeuronId]:
        raise NotImplementedError

    def get_covered_neurons(self) -> Set[NeuronId]:
        raise NotImplementedError

    def has_observations(self) -> bool:
        """Whether update() has ever run. Distinguishes "no masks built yet"
        from "fully covered", which get_uncovered_neurons() reports alike."""
        raise NotImplementedError


def _activation_to_neuron_matrix(activation: torch.Tensor) -> torch.Tensor:
    """
    Convert (N,...) activation tensor to (N, neurons) matrix for coverage.
    
    - (N, neurons) [dim==2]: already correct → return as-is
    - (N, C, H, W) [dim==4]: max-pool spatial dims → (N, C)
    - Other: flatten all but batch dim → (N, K)
    """
    if activation.dim() == 2:
        return activation  # (N, neurons)
    if activation.dim() == 4:
        return activation.abs().amax(dim=(2, 3))  # (N, C)
    if activation.dim() == 1:
        return activation.unsqueeze(1)  # (N, 1)
    return activation.flatten(start_dim=1)  # (N, K)


class BestInputCov(CoverageStrategy):
    """
    Per-input neuron coverage.

    - Each update() computes coverage for that specific input only.
    - get_coverage() returns the best (max) per-input coverage seen so far (monotonic).
    """

    def __init__(self, model: nn.Module, threshold: float = 0.1):
        super().__init__(model, threshold)
        self._layer_neuron_counts: Dict[str, int] = {}
        self._inputs_seen: int = 0
        self._sum_coverage: float = 0.0
        self.last_input_coverage: float = 0.0
        self.best_input_coverage: float = 0.0

    def update(
        self, input_tensor: torch.Tensor, activations: Dict[str, torch.Tensor]
    ) -> Tuple[float, torch.Tensor]:
        """
        Update per-input neuron coverage with batch activations.
        
        Batch-native: computes coverage for all N samples in one vectorized call.
        A sample is "interesting" if its individual coverage exceeds the pre-batch best.
        
        Returns:
            (global_delta, interesting_mask):
            - global_delta: improvement in best_input_coverage (float)
            - interesting_mask: BoolTensor (N,), True for samples exceeding previous best
        """
        for layer_name, activation in activations.items():
            mat = _activation_to_neuron_matrix(activation)
            if mat.numel() == 0:
                continue
            self._layer_neuron_counts.setdefault(layer_name, mat.shape[1])
        
        total_neurons = int(sum(self._layer_neuron_counts.values()))
        if total_neurons == 0:
            N = input_tensor.shape[0]
            return 0.0, torch.zeros(N, dtype=torch.bool)
        
        old_best = float(self.best_input_coverage)
        N = input_tensor.shape[0]
        
        total_fired = torch.zeros(N)
        for layer_name, activation in activations.items():
            mat = _activation_to_neuron_matrix(activation)
            if mat.numel() == 0:
                continue
            total_fired += (mat > float(self.threshold)).sum(dim=1).float()
        
        sample_covs = total_fired / total_neurons  # (N,)
        
        interesting_mask = sample_covs > old_best
        
        # Running stats — single .sum()/.max() reductions, no per-sample .tolist()
        self._inputs_seen += N
        self._sum_coverage += float(sample_covs.sum())
        self.last_input_coverage = float(sample_covs[-1])
        batch_best = float(sample_covs.max())
        if batch_best > self.best_input_coverage:
            self.best_input_coverage = batch_best
        
        global_delta = max(0.0, float(self.best_input_coverage) - old_best)
        return global_delta, interesting_mask

    def get_coverage(self) -> float:
        return float(self.best_input_coverage)

    def get_stats(self) -> Dict[str, Any]:
        avg = (self._sum_coverage / self._inputs_seen) if self._inputs_seen else 0.0
        return {
            "coverage": float(self.get_coverage()),
            "inputs_seen": self._inputs_seen,
            "last_input_coverage": float(self.last_input_coverage),
            "best_input_coverage": float(self.best_input_coverage),
            "avg_input_coverage": float(avg),
            "total_neurons_seen": int(sum(self._layer_neuron_counts.values())),
            "layers_seen": int(len(self._layer_neuron_counts)),
        }

    def has_observations(self) -> bool:
        return self._inputs_seen > 0

    def reset(self) -> None:
        self._layer_neuron_counts.clear()
        self._inputs_seen = 0
        self._sum_coverage = 0.0
        self.last_input_coverage = 0.0
        self.best_input_coverage = 0.0


class GlobalCov(CoverageStrategy):
    """
    Global union neuron coverage.

    A neuron is covered if it has fired at least once across all inputs.
    Coverage state is stored as per-layer BoolTensors on the same device as
    activations, enabling O(1) vectorized lookup instead of Python set iteration.
    """

    def __init__(self, model: nn.Module, threshold: float = 0.1,
                 per_instance: int = 0):
        super().__init__(model, threshold)

        self._layer_neuron_counts: Dict[str, int] = {}
        self._covered_masks: Dict[str, torch.Tensor] = {}
        self.last_newly_covered_count: int = 0
        # per_instance > 0 keeps ONE coverage row per verification instance
        # instead of a single union over the whole batch.
        #
        # Why this is a real choice and not a detail: with a single union, a
        # sample counts as interesting only if it fires a neuron NO instance
        # has ever fired. Instance A firing a neuron in iteration 3 silently
        # raises the bar for the other 99 instances for the rest of the run,
        # and the bar keeps rising as the batch explores -- so the same
        # benchmark gets a harsher admission rule purely for being run in a
        # bigger batch. At B=1 the two modes are identical, which is why this
        # only shows up once batching exists. Measured consequence of the
        # union rule: the coverage arm's corpus stalls at ~273 live seeds
        # against ~3010 for state-based admission.
        #
        # State admission (PatternStateManager) is already per instance --
        # its fingerprints carry original_index -- so this makes the coverage
        # path agree with it rather than compete on a different scope.
        self._per_instance = int(per_instance)

    def _ensure_layer_registered(
        self, layer_name: str, neuron_count: int, device: torch.device
    ) -> None:
        neuron_count = int(neuron_count)
        if neuron_count <= 0:
            return
        if layer_name in self._layer_neuron_counts:
            return
        self._layer_neuron_counts[layer_name] = neuron_count
        shape = ((self._per_instance, neuron_count) if self._per_instance > 0
                 else (neuron_count,))
        self._covered_masks[layer_name] = torch.zeros(
            shape, dtype=torch.bool, device=device
        )

    def update(
        self, input_tensor: torch.Tensor, activations: Dict[str, torch.Tensor],
        rows: Optional[torch.Tensor] = None,
    ) -> Tuple[float, torch.Tensor]:
        """
        Update global union neuron coverage with batch activations.
        
        Batch-native: processes all N samples in one call. A sample is "interesting"
        if it fires any neuron not covered before this batch (AFL-style: any input
        hitting a new edge is interesting, multiple samples can share credit).
        
        All tensor operations stay on input_tensor.device 
        
        Returns:
            (global_delta, interesting_mask):
            - global_delta: fraction of newly covered neurons (float)
            - interesting_mask: BoolTensor (N,), True for samples that fire
              at least one previously-uncovered neuron
        """
        N = input_tensor.shape[0]
        old_covered = self._total_covered()
        interesting_mask = torch.zeros(N, dtype=torch.bool)
        
        for layer_name, activation in activations.items():
            mat = _activation_to_neuron_matrix(activation)  # (N, neurons)
            if mat.numel() == 0:
                continue
            n_neurons = mat.shape[1]
            self._ensure_layer_registered(layer_name, n_neurons, device=mat.device)
            
            fired_mask = mat > float(self.threshold)  # (N, neurons)
            already_covered = self._covered_masks[layer_name]

            if self._per_instance > 0 and rows is None:
                # The mask is [instances, neurons] here; the union path below
                # would broadcast it against [N, neurons] and silently produce
                # nonsense. A caller that scopes coverage per instance has to
                # say which instance each lane belongs to.
                raise ValueError(
                    "GlobalCov(per_instance>0).update() needs rows= (one "
                    "instance index per lane); got None."
                )
            if self._per_instance > 0:
                idx = rows.to(already_covered.device).long()
                # Each lane is scored against ITS OWN instance's row.
                already_lane = already_covered[idx]                    # (N, neurons)
                newly_lane = fired_mask & ~already_lane
                interesting_mask |= newly_lane.any(dim=1).cpu()
                # index_add_ on an integer accumulator, not index_put_: the
                # corpus samples with replacement, so several lanes can carry
                # the SAME instance and index_put_ would keep only one of them.
                acc = torch.zeros_like(already_covered, dtype=torch.uint8)
                acc.index_add_(0, idx, fired_mask.to(torch.uint8))
                self._covered_masks[layer_name] = already_covered | (acc > 0)
                continue

            fired_any = fired_mask.any(dim=0)  # (neurons,)
            newly_covered = fired_any & ~already_covered  # (neurons,)

            if newly_covered.any():
                # (N, neurons) & (1, neurons) → (N, neurons) → any(dim=1) → (N,)
                interesting_mask |= (fired_mask & newly_covered.unsqueeze(0)).any(dim=1)

            # Update coverage mask in-place (bitwise OR)
            self._covered_masks[layer_name] = already_covered | fired_any
        
        new_covered = self._total_covered()
        self.last_newly_covered_count = new_covered - old_covered
        total_neurons = self._total_neurons()
        global_delta = (self.last_newly_covered_count / total_neurons) if total_neurons > 0 else 0.0
        return global_delta, interesting_mask

    def _total_covered(self) -> int:
        """Union over instances, always -- the REPORTED coverage must stay
        comparable across arms, so per-instance mode changes what counts as
        interesting, not what gets reported."""
        return sum(int((m.any(dim=0) if m.dim() > 1 else m).sum())
                   for m in self._covered_masks.values())

    def _total_neurons(self) -> int:
        return sum(self._layer_neuron_counts.values())

    def get_coverage(self) -> float:
        total = self._total_neurons()
        if total == 0:
            return 0.0
        return self._total_covered() / total

    def get_uncovered_neurons(self) -> Set[NeuronId]:
        result: Set[NeuronId] = set()
        for layer_name, mask in self._covered_masks.items():
            # Union over instances in per-instance mode: "uncovered" for
            # reporting (and for HPGD-Cov's target pool) means no instance has
            # fired it, not "this instance has not".
            mask = mask.any(dim=0) if mask.dim() > 1 else mask
            for idx in (~mask).nonzero(as_tuple=True)[0].tolist():
                result.add((layer_name, int(idx)))
        return result

    def get_covered_neurons(self) -> Set[NeuronId]:
        result: Set[NeuronId] = set()
        for layer_name, mask in self._covered_masks.items():
            mask = mask.any(dim=0) if mask.dim() > 1 else mask
            for idx in mask.nonzero(as_tuple=True)[0].tolist():
                result.add((layer_name, int(idx)))
        return result

    def get_stats(self) -> Dict[str, Any]:
        return {
            "coverage": float(self.get_coverage()),
            "covered_neurons": self._total_covered(),
            "total_neurons": self._total_neurons(),
            "last_newly_covered": int(self.last_newly_covered_count),
            "layers_seen": int(len(self._layer_neuron_counts)),
        }

    def has_observations(self) -> bool:
        return bool(self._covered_masks)

    def reset(self) -> None:
        self._covered_masks.clear()
        self._layer_neuron_counts.clear()
        self.last_newly_covered_count = 0

class CoverageTracker:
    """
    Coverage engine that delegates to coverage strategies.
    
    Supports runtime strategy switching via update(strategy=...).
    Strategies are lazily initialized on first use.
    """

    _REGISTRY = {"BestInputCov": BestInputCov, "GlobalCov": GlobalCov}

    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.1,
        strategy: str = "BestInputCov",
        per_instance: int = 0,
    ):
        self.model = model
        self.threshold = threshold
        self.strategy = strategy
        # >0 scopes GlobalCov's "already covered" per verification instance
        # instead of one union over the batch; 0 (default) is the original
        # union. Only GlobalCov honours it -- BestInputCov is per INPUT, a
        # different axis, and keeps no union at all.
        self.per_instance = int(per_instance)
        self._strategies: Dict[str, CoverageStrategy] = {}
        
        if strategy not in self._REGISTRY:
            raise ValueError(f"Unknown coverage strategy '{strategy}'. Valid: {list(self._REGISTRY.keys())}")

    def _get_strategy(self, name: str) -> CoverageStrategy:
        """Get or create strategy by name (lazy init)."""
        if name not in self._strategies:
            if name not in self._REGISTRY:
                raise ValueError(f"Unknown coverage strategy '{name}'. Valid: {list(self._REGISTRY.keys())}")
            kwargs = {"model": self.model, "threshold": self.threshold}
            if name == "GlobalCov" and self.per_instance > 0:
                kwargs["per_instance"] = self.per_instance
            self._strategies[name] = self._REGISTRY[name](**kwargs)
        return self._strategies[name]

    def update(
        self,
        input_tensor: torch.Tensor,
        activations: Dict[str, torch.Tensor],
        strategy: Optional[str] = None,
        rows: Optional[torch.Tensor] = None,
    ) -> Tuple[float, torch.Tensor]:
        """
        Update coverage with activations from a fuzzing iteration.
        
        Batch-native: processes all N samples in one vectorized call.
        Called by ACTFuzzer after each inference.
        
        Args:
            input_tensor: Mutated input batch [N, ...]
            activations: Dict of layer activations from MutationEngine hooks
            strategy: Override default strategy for this call (optional)
        
        Returns:
            (global_delta, interesting_mask):
            - global_delta: overall coverage improvement (float, 0..1)
            - interesting_mask: BoolTensor (N,), True for interesting samples
        """
        s = strategy if strategy is not None else self.strategy
        strat = self._get_strategy(s)
        # Only GlobalCov can scope by instance; BestInputCov is per INPUT, a
        # different axis, and its update() takes no rows.
        if rows is not None and isinstance(strat, GlobalCov):
            return strat.update(input_tensor, activations, rows=rows)
        return strat.update(input_tensor, activations)

    def get_coverage(self, strategy: Optional[str] = None) -> float:
        s = strategy if strategy is not None else self.strategy
        return self._get_strategy(s).get_coverage()

    def get_stats(self, strategy: Optional[str] = None) -> Dict[str, Any]:
        s = strategy if strategy is not None else self.strategy
        return {"strategy": s, **self._get_strategy(s).get_stats()}

    def has_observations(self, strategy: Optional[str] = None) -> bool:
        """Whether update() has ever run for this strategy.

        get_uncovered_neurons() returns an empty set for BOTH "nothing has been
        observed yet" (masks not built) and "everything is covered" -- opposite
        situations that a caller steering on uncovered neurons must not
        conflate. Consult this first; see HPGDCoverageMutation.mutate.
        """
        s = strategy if strategy is not None else self.strategy
        if s not in self._strategies:
            return False
        return self._strategies[s].has_observations()

    def get_uncovered_neurons(self, strategy: Optional[str] = None) -> Set[NeuronId]:
        s = strategy if strategy is not None else self.strategy
        return self._get_strategy(s).get_uncovered_neurons()

    def get_covered_neurons(self, strategy: Optional[str] = None) -> Set[NeuronId]:
        s = strategy if strategy is not None else self.strategy
        return self._get_strategy(s).get_covered_neurons()

