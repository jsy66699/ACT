"""
PatternStateManager: global ReLU-sign-pattern state tracking shared across
every mutation strategy (not just HPGD).

Replaces CoverageTracker-driven corpus admission with a real state-dedup /
diversity check, and SeedCorpus's pure energy-weighted scheduling with a
density-aware one -- both restricted to the instance's UNSTABLE ReLU
neurons (stable neurons can never change sign for any input in the box, so
they carry zero information for either dedup or scheduling).

Three pieces, each solving a different one of "efficient dedup", "spread
new states out", and "bias exploitation toward neurons that just proved
flippable from this seed":

  - `compute_unstable_mask`: interval bound propagation via the existing
    act.back_end verifier (no new bound-propagation implementation).
  - `_BloomFilter`: fixed-memory, O(1)-ish exact-novelty pre-check (replaces
    the O(N)-memory-per-entry hash set from the first HPGD version).
  - `_StateBKTree`: Hamming-distance diversity gate over the unstable
    subspace, whose incrementally-maintained per-node subtree size doubles
    as an anchor-relative local-density estimate for scheduling (a sparse
    subtree size means "few points have ever routed near this one").

`PatternStateManager` ties these together and additionally tracks, per
seed (keyed by a hash of its own input tensor), a `local_bias` weight
vector over the unstable candidate set: high weight on neurons that were
just observed to flip producing that seed, low background weight on the
rest -- read by HPGDMutation to bias its next K-neuron selection FROM that
seed toward exploiting a locally-proven-flippable direction, without
collapsing exploration to zero elsewhere.

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from act.pipeline.fuzzing.corpus import FuzzingSeed


# ---------------------------------------------------------------------------
# Unstable-ReLU precompute (bound propagation via the existing back_end)
# ---------------------------------------------------------------------------


def compute_unstable_mask(
    wrapped_model: nn.Module,
    lb: torch.Tensor,
    ub: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Boolean mask [N] over every ReLU pre-activation neuron, True where the
    neuron is UNSTABLE (pre-activation interval straddles zero for some
    input in the box [lb, ub]), in the same flattened order as
    act.pipeline.fuzzing.mutations._relu_preactivations_batched.

    Computed via interval bound propagation through the existing
    act.back_end verifier (TorchToACT + analyze()) -- the same
    ``(lb < 0) & (ub > 0)`` test used by the BaB branching code
    (act/back_end/solver/solver_dual.py) and (act/back_end/bab/branching).

    Returns None (== "treat every neuron as a candidate") if bound
    propagation fails for any reason, or if the per-layer neuron count
    derived from the Net graph doesn't match a forward-hook pass over the
    same model -- a correctness guard against silently relying on the
    act.back_end Net's layer ordering matching nn.Module.modules() order
    for architectures where that isn't guaranteed.
    """
    try:
        from act.back_end.analyze import analyze
        from act.back_end.core import Bounds, ConSet, Fact
        from act.back_end.layer_schema import LayerKind
        from act.back_end.verifier import find_entry_layer_id
        from act.pipeline.verification.torch2act import TorchToACT
    except Exception:
        return None

    try:
        net = TorchToACT(wrapped_model).run()
        entry_id = find_entry_layer_id(net)
        entry_fact = Fact(bounds=Bounds(lb.clone(), ub.clone()), cons=ConSet())
        before, _after, _globalC = analyze(net, entry_id, entry_fact)

        relu_masks: List[torch.Tensor] = []
        for layer in net.layers:
            kind = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
            if kind != LayerKind.RELU.value:
                continue
            bounds = before[layer.id].bounds
            layer_lb = bounds.lb.flatten(start_dim=1)
            layer_ub = bounds.ub.flatten(start_dim=1)
            unstable = (layer_lb < 0) & (layer_ub > 0)
            relu_masks.append(unstable[0])
        if not relu_masks:
            return None
        mask = torch.cat(relu_masks, dim=0)
    except Exception:
        return None

    try:
        from act.pipeline.fuzzing.mutations import _relu_preactivations_batched

        with torch.no_grad():
            sample = lb + torch.rand_like(lb) * (ub - lb).clamp(min=0)
            z = _relu_preactivations_batched(wrapped_model, sample)
        if z.shape[1] != mask.shape[0]:
            return None
    except Exception:
        return None

    return mask.to(device=lb.device)


# ---------------------------------------------------------------------------
# Bloom filter: fixed-memory approximate "have I seen this exact state"
# ---------------------------------------------------------------------------


class _BloomFilter:
    """Fixed-memory approximate membership set for exact-pattern novelty
    checks. False positives are possible (tunable via `num_bits`/
    `num_hashes`); false negatives are not, so a miss here is a hard
    guarantee of novelty. Memory is O(num_bits) regardless of how many
    patterns have been inserted -- unlike a hash set of raw pattern bytes,
    which grows O(num_states x N).

    Hashing still round-trips each pattern through CPU/numpy once per call
    (`hash(bytes)`); a fully GPU-resident hash (e.g. a fixed random
    projection + modulo, computed as a tensor op) would remove that, but is
    left as a follow-up -- this already turns an O(N)-per-entry-growing
    structure into an O(1)-memory one, which is the win that matters first.
    """

    def __init__(self, num_bits: int = 1 << 20, num_hashes: int = 4, device: Optional[torch.device] = None):
        self.num_bits = int(num_bits)
        self.num_hashes = max(1, int(num_hashes))
        self.bits = torch.zeros(self.num_bits, dtype=torch.bool, device=device)

    def _bit_indices(self, pattern: torch.Tensor) -> List[int]:
        key = pattern.to(torch.int8).cpu().numpy().tobytes()
        h1 = hash(key) % self.num_bits
        h2 = (hash((key, 1)) % (self.num_bits - 1)) + 1
        return [(h1 + i * h2) % self.num_bits for i in range(self.num_hashes)]

    def might_contain(self, pattern: torch.Tensor) -> bool:
        idx = self._bit_indices(pattern)
        return bool(self.bits[idx].all().item())

    def add(self, pattern: torch.Tensor) -> None:
        idx = self._bit_indices(pattern)
        self.bits[idx] = True


# ---------------------------------------------------------------------------
# BK-tree over the unstable subspace, with incremental density tracking
# ---------------------------------------------------------------------------


@dataclass
class _BKNode:
    point: torch.Tensor
    payload: Dict[str, Any]
    children: Dict[int, "_BKNode"] = field(default_factory=dict)
    subtree_size: int = 1


def _hamming(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((a != b).sum().item())


class _StateBKTree:
    """BK-tree over Hamming distance, restricted to the unstable-neuron
    subspace.

    Beyond the diversity-threshold membership query (`has_within`), every
    node's `subtree_size` is incremented along the full root-to-insertion
    path on every insert, giving each existing node an always-current,
    incrementally-maintained (no full-tree recompute) local density
    estimate: a node with a small subtree_size has had few points route
    near it so far. This is anchor-relative and insertion-order dependent
    (BK-trees don't rebalance) rather than a true spatial density map, but
    it is cheap and good enough to bias scheduling toward sparser regions.
    """

    def __init__(self) -> None:
        self._root: Optional[_BKNode] = None
        self.size = 0

    def has_within(self, point: torch.Tensor, threshold: int) -> bool:
        if self._root is None:
            return False
        stack = [self._root]
        while stack:
            node = stack.pop()
            d = _hamming(node.point, point)
            if d <= threshold:
                return True
            for dist, child in node.children.items():
                if abs(dist - d) <= threshold:
                    stack.append(child)
        return False

    def insert(self, point: torch.Tensor, payload: Dict[str, Any]) -> _BKNode:
        if self._root is None:
            self._root = _BKNode(point=point, payload=payload)
            self.size += 1
            return self._root
        node = self._root
        while True:
            d = _hamming(node.point, point)
            node.subtree_size += 1
            if d == 0:
                return node  # exact duplicate of an existing node
            if d in node.children:
                node = node.children[d]
            else:
                child = _BKNode(point=point, payload=payload)
                node.children[d] = child
                self.size += 1
                return child


# ---------------------------------------------------------------------------
# PatternStateManager
# ---------------------------------------------------------------------------


class PatternStateManager:
    """Global state manager shared by every mutation strategy.

    `observe()` is the single admission decision for corpus membership when
    the fuzzer's `admission_mode` is "state": a mutated sample is admitted
    iff its achieved pattern (restricted to unstable neurons) is novel --
    Bloom-filter miss, then a BK-tree diversity-threshold check (rejects if
    within `diversity_threshold - 1` of anything already recorded).
    Admitted samples are inserted into the BK-tree and become eligible for
    `pick_seeds` (density-weighted, optionally energy-blended scheduling)
    and, if they were counterexamples, `pick_ce_seeds` (anchors for the GCE
    pull-back task).
    """

    def __init__(
        self,
        unstable_mask: Optional[torch.Tensor],
        total_neurons: int,
        diversity_threshold: int = 1,
        bloom_bits: int = 1 << 20,
        bloom_hashes: int = 4,
        local_bias_high: float = 10.0,
        local_bias_low: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        if unstable_mask is not None:
            self.candidate_indices = unstable_mask.nonzero(as_tuple=True)[0].to(device)
        else:
            self.candidate_indices = torch.arange(total_neurons, device=device)
        self.num_candidates = int(self.candidate_indices.numel())
        self.diversity_threshold = int(diversity_threshold)
        self.bloom = _BloomFilter(bloom_bits, bloom_hashes, device=device)
        self.tree = _StateBKTree()
        self.local_bias_high = float(local_bias_high)
        self.local_bias_low = float(local_bias_low)
        self.device = device

        self._registry: List[_BKNode] = []
        self._local_bias: Dict[bytes, torch.Tensor] = {}
        self._ce_seeds: List[Dict[str, torch.Tensor]] = []

    # -- helpers ------------------------------------------------------------

    def restrict(self, pattern_full: torch.Tensor) -> torch.Tensor:
        """Restrict a full [..., N] pattern to the unstable candidate dims."""
        return pattern_full.index_select(-1, self.candidate_indices)

    @staticmethod
    def hash_seed(seed_tensor: torch.Tensor) -> bytes:
        return seed_tensor.detach().to(torch.float32).cpu().numpy().tobytes()

    # -- admission / insertion -----------------------------------------------

    def observe(
        self,
        seed_tensor: torch.Tensor,
        pattern_full: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        original_tensor: Optional[torch.Tensor] = None,
        original_index: Optional[torch.Tensor] = None,
        is_ce: bool = False,
        energy_bonus: float = 1.0,
    ) -> bool:
        """seed_tensor: [1, ...]; pattern_full: [1, N] (or [N]) full ReLU
        sign pattern, unrestricted. `label`/`original_tensor`/
        `original_index` are stored alongside the seed (defaulting to the
        seed itself / index 0 / no label) so `pick_seeds`/`pick_ce_seeds`
        can hand back everything needed to rebuild a proper FuzzingSeed
        batch later. Returns True iff admitted (novel/diverse enough), in
        which case the state is recorded (tree + registry, and the
        CE-anchor pool if `is_ce`)."""
        pattern = self.restrict(pattern_full.reshape(-1))

        if not self.bloom.might_contain(pattern):
            self.bloom.add(pattern)
        # A bloom hit can be a false positive, so the BK-tree check below
        # still runs regardless -- the bloom filter only saves the (rarer)
        # true-miss case from paying for a tree traversal at all... it
        # doesn't currently short-circuit anything by itself yet since the
        # tree check is cheap relative to the hash; kept as a structural
        # hook for a future GPU-resident hash where the bloom check really
        # is far cheaper than a tree walk.

        reject_threshold = max(0, self.diversity_threshold - 1)
        if self.tree.has_within(pattern, reject_threshold):
            return False

        payload = {
            "seed": seed_tensor.detach().clone(),
            "label": (label.detach().clone() if label is not None else torch.full((1,), -1, dtype=torch.long)),
            "original_tensor": (
                original_tensor.detach().clone() if original_tensor is not None else seed_tensor.detach().clone()
            ),
            "original_index": (
                original_index.detach().clone() if original_index is not None
                else torch.zeros(1, dtype=torch.long)
            ),
            "is_ce": bool(is_ce),
            "energy_bonus": float(energy_bonus),
        }
        node = self.tree.insert(pattern, payload=payload)
        self._registry.append(node)
        if is_ce:
            self._ce_seeds.append({**payload, "pattern": pattern.detach().clone()})
        return True

    def update_local_bias(
        self,
        new_seed: torch.Tensor,
        flipped_candidate_positions: torch.Tensor,
    ) -> None:
        """Record which unstable-subspace positions actually flipped
        producing `new_seed`, so a future HPGD call FROM `new_seed` biases
        K-neuron selection toward those positions (exploitation), with a
        small background weight over the rest (exploration).
        `flipped_candidate_positions` are indices into `self.candidate_indices`
        (i.e. already restricted-space indices, not raw neuron ids)."""
        weights = torch.full((self.num_candidates,), self.local_bias_low, device=self.device)
        if flipped_candidate_positions.numel() > 0:
            weights[flipped_candidate_positions] = self.local_bias_high
        self._local_bias[self.hash_seed(new_seed)] = weights

    def local_bias(self, seed_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        return self._local_bias.get(self.hash_seed(seed_tensor))

    # -- scheduling -----------------------------------------------------------

    def pick_seeds(self, n: int, use_energy: bool = True) -> List[Dict[str, torch.Tensor]]:
        """Density-weighted (sparser subtree = higher weight) seed sampling
        for the BI (explore) task, optionally blended with each seed's
        energy_bonus (e.g. a violation multiplier) so already-interesting
        regions aren't starved purely for being locally dense. Returns the
        stored payload dicts (seed/label/original_tensor/original_index);
        pass them to `seeds_to_batch` to get a FuzzingSeed."""
        if not self._registry:
            return []
        weights = []
        for node in self._registry:
            sparsity = 1.0 / (1.0 + float(node.subtree_size))
            bonus = node.payload.get("energy_bonus", 1.0) if use_energy else 1.0
            weights.append(sparsity * bonus)
        chosen = random.choices(self._registry, weights=weights, k=n)
        return [node.payload for node in chosen]

    def pick_ce_seeds(self, n: int) -> List[Dict[str, torch.Tensor]]:
        """Anchors for the GCE pull-back task: payload dicts (including
        "pattern", the anchor's own full ReLU sign pattern restricted to
        unstable dims) sampled from confirmed counterexamples. Empty if none
        have been found yet (GCE should idle until BI or a coverage-mode
        strategy finds one)."""
        if not self._ce_seeds:
            return []
        return random.choices(self._ce_seeds, k=n)

    @staticmethod
    def seeds_to_batch(payloads: List[Dict[str, torch.Tensor]]) -> "FuzzingSeed":
        """Assemble a list of observe()-stored payload dicts into a batched
        FuzzingSeed, the same shape MutationEngine.mutate()/HPGDPullbackMutation
        expect as input."""
        from act.pipeline.fuzzing.corpus import FuzzingSeed

        return FuzzingSeed(
            tensor=torch.cat([p["seed"] for p in payloads], dim=0),
            original_tensor=torch.cat([p["original_tensor"] for p in payloads], dim=0),
            original_index=torch.cat([p["original_index"] for p in payloads], dim=0),
            label=torch.cat([p["label"] for p in payloads], dim=0),
        )

    def __len__(self) -> int:
        return len(self._registry)
