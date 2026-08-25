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
        # Exact-match index, maintained alongside the tree. At threshold 0 --
        # which is what diversity_threshold=1, the default, reduces to -- the
        # tree walk below is answering "has this exact pattern been seen", and
        # a walk is the wrong structure for that: every miss visits the whole
        # tree, each visit paying a _hamming() .item() sync, and admission runs
        # ~97% on a real benchmark so misses are the common case. Profiling a
        # 20s safenlp run: has_within 3.04s of 20s total. This dict answers the
        # same question in O(1) and is only consulted when threshold == 0, so
        # the tree stays authoritative for every diversity_threshold > 1.
        self._exact: set = set()

    @staticmethod
    def _key(point: torch.Tensor) -> bytes:
        return point.detach().to(torch.int8).cpu().numpy().tobytes()

    def has_within(self, point: torch.Tensor, threshold: int) -> bool:
        if self._root is None:
            return False
        if threshold == 0:
            return self._key(point) in self._exact
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
        self._exact.add(self._key(point))
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

        # Fingerprint fast path, used by observe_batch when the diversity
        # threshold reduces to exact-match. A restricted pattern is a bit
        # vector, so packing it into a few int64 words is lossless: two
        # patterns share a fingerprint iff they agree on every candidate
        # neuron. That turns "compare 128 dims against every stored pattern"
        # into "compare 4 integers", making lookup cost independent of how
        # many patterns are stored -- which is the whole problem, since the
        # registry grows past 100k within a 60s run.
        # 32 bits per word (not 64) keeps every partial product well inside
        # signed int64, so the packing matmul cannot overflow.
        self._BITS_PER_WORD = 32
        self._pack_words = (self.num_candidates + self._BITS_PER_WORD - 1) // self._BITS_PER_WORD
        self._pow2 = (2 ** torch.arange(self._BITS_PER_WORD, dtype=torch.int64, device=device))
        self._seen: set = set()

        # Per-neuron marginal occupancy over admitted patterns, for steering
        # flips toward under-explored regions. The exact question -- "which
        # target is farthest from everything seen" -- is O(B*M*C) per iteration
        # against a registry that passes 100k, so this tracks the marginal
        # instead: how often each candidate neuron has been +1. That ignores
        # correlations between neurons, but costs one sum per batch and one
        # compare per lane rather than a nearest-neighbour scan.
        self._sign_count = torch.zeros(self.num_candidates, dtype=torch.float64, device=device)
        self._sign_total = 0

    # -- helpers ------------------------------------------------------------

    def restrict(self, pattern_full: torch.Tensor) -> torch.Tensor:
        """Restrict a full [..., N] pattern to the unstable candidate dims."""
        return pattern_full.index_select(-1, self.candidate_indices)

    def fingerprint(self, restricted: torch.Tensor) -> torch.Tensor:
        """Pack [B, C] of +-1 into [B, W] int64, losslessly, in one matmul."""
        B, C = restricted.shape
        bits = (restricted > 0).to(torch.int64)
        pad = self._pack_words * self._BITS_PER_WORD - C
        if pad:
            bits = torch.cat([bits, bits.new_zeros(B, pad)], dim=1)
        return (bits.view(B, self._pack_words, self._BITS_PER_WORD) * self._pow2).sum(dim=2)

    def seen_mask(
        self,
        patterns_full: torch.Tensor,
        original_indices: torch.Tensor,
        per_instance: bool = True,
    ) -> torch.Tensor:
        """[B] bool: which of these patterns are already recorded. Pure query,
        records nothing -- for asking "would this have been rejected" without
        perturbing the run, e.g. measuring how often HPGD aims at a state it
        has already visited."""
        B = patterns_full.shape[0]
        keys = self.fingerprint(self.restrict(patterns_full.reshape(B, -1)))
        if per_instance:
            keys = torch.cat([original_indices.reshape(B, 1).to(keys.device), keys], dim=1)
        keys_cpu = keys.cpu().numpy()
        return torch.tensor([keys_cpu[b].tobytes() in self._seen for b in range(B)],
                            dtype=torch.bool)

    def observe_batch(
        self,
        seed_tensors: torch.Tensor,
        patterns_full: torch.Tensor,
        labels: Optional[torch.Tensor],
        original_tensors: torch.Tensor,
        original_indices: torch.Tensor,
        is_ce_mask: torch.Tensor,
        per_instance: bool = True,
    ) -> torch.Tensor:
        """Exact-match admission for a whole batch. Returns BoolTensor[B].

        Only valid when diversity_threshold reduces to exact match -- a
        fingerprint proves equality but says nothing about Hamming distance,
        so any larger threshold must keep using observe()/the BK-tree.

        With per_instance=True the instance index joins the fingerprint, so
        each verification problem keeps its own state space in one shared
        table. That matters because a batch lane is a DIFFERENT instance with
        its own input box and property: lane 300 reproducing a pattern lane 5
        already logged is novel FOR LANE 300, and the global table rejected it.
        """
        B = seed_tensors.shape[0]
        restricted = self.restrict(patterns_full.reshape(B, -1))
        keys = self.fingerprint(restricted)
        if per_instance:
            keys = torch.cat([original_indices.reshape(B, 1).to(keys.device), keys], dim=1)

        # One transfer for the batch, instead of per-sample slicing and .item().
        keys_cpu = keys.cpu().numpy()
        is_ce_list = is_ce_mask.tolist()
        admitted = torch.zeros(B, dtype=torch.bool, device=seed_tensors.device)
        admitted_list = []
        for b in range(B):
            key = keys_cpu[b].tobytes()
            if key in self._seen:
                continue
            self._seen.add(key)
            admitted[b] = True
            admitted_list.append(b)

        # Marginal occupancy counts only what was actually admitted -- rejected
        # duplicates would double-count states already represented.
        if admitted_list:
            adm = restricted[admitted_list]
            self._sign_count += (adm > 0).sum(dim=0).to(self._sign_count.dtype)
            self._sign_total += len(admitted_list)

        # Payloads are only materialized for admitted lanes, and only when
        # something can actually consume them (pick_seeds / pick_ce_seeds).
        for b in admitted_list:
            is_ce = bool(is_ce_list[b])
            payload = {
                "seed": seed_tensors[b : b + 1].detach().clone(),
                "label": (labels[b : b + 1].detach().clone() if labels is not None
                          else torch.full((1,), -1, dtype=torch.long)),
                "original_tensor": original_tensors[b : b + 1].detach().clone(),
                "original_index": original_indices[b : b + 1].detach().clone(),
                "is_ce": is_ce,
                "energy_bonus": 5.0 if is_ce else 1.0,
            }
            node = _BKNode(point=restricted[b].detach().clone(), payload=payload)
            self._registry.append(node)
            self.tree.size += 1
            if is_ce:
                self._ce_seeds.append({**payload, "pattern": restricted[b].detach().clone()})
        return admitted

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

        reject_threshold = max(0, self.diversity_threshold - 1)

        # The bloom filter is a pre-check for the tree walk, so it only earns
        # its keep when there IS a walk. At reject_threshold 0 the tree query
        # is already an O(1) exact-match lookup (see _StateBKTree._exact), so
        # hashing the pattern here would cost strictly more than the check it
        # is meant to save -- profiling a 20s safenlp run put might_contain at
        # 1.58s of 20s while short-circuiting nothing, because a bloom hit can
        # be a false positive and the tree check has to run regardless.
        if reject_threshold > 0 and not self.bloom.might_contain(pattern):
            self.bloom.add(pattern)

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

    def sparsity_weights(self, natural_restricted: torch.Tensor) -> torch.Tensor:
        """[B, C] flip scores: high where flipping LEAVES a crowded side.

        For lane b and candidate j, flipping sends neuron j to the sign
        opposite its current one, so the score is 1 - freq(destination sign):
        currently +1 scores freq_plus (its -1 destination is rare exactly when
        +1 is common), currently -1 scores 1 - freq_plus. A neuron already
        sitting in the rare state therefore scores low -- flipping it back into
        the crowd is the opposite of what this is for.

        Uniform until something has been admitted, so an empty history steers
        nothing rather than steering arbitrarily.
        """
        if self._sign_total == 0:
            return torch.ones_like(natural_restricted, dtype=torch.float32)
        freq_plus = (self._sign_count / self._sign_total).to(
            natural_restricted.device, torch.float32
        )
        return torch.where(natural_restricted > 0, freq_plus, 1.0 - freq_plus).clamp(min=1e-3)

    def local_bias_batch(self, seed_tensors: torch.Tensor) -> torch.Tensor:
        """[B, num_candidates] flip weights for a whole batch of seeds.

        Seeds with no recorded bias (never admitted, or admitted before this
        manager saw them) fall back to a uniform row, which reproduces HPGD's
        unbiased torch.randperm choice for that lane -- so an unknown seed
        behaves exactly as it did before this was wired up, rather than being
        silently steered by someone else's history.

        One host transfer for the batch, then per-lane dict lookups, mirroring
        observe_batch: hash_seed keys on raw float32 bytes, so the lookup has
        to happen lane by lane regardless.
        """
        B = seed_tensors.shape[0]
        flat = seed_tensors.detach().to(torch.float32).cpu().numpy().reshape(B, -1)
        rows = []
        for b in range(B):
            w = self._local_bias.get(flat[b].tobytes())
            rows.append(w if w is not None
                        else torch.full((self.num_candidates,), 1.0, device=self.device))
        return torch.stack(rows, dim=0)

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
