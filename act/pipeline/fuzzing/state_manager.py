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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from act.pipeline.fuzzing.corpus import FuzzingSeed


# ---------------------------------------------------------------------------
# Unstable-ReLU precompute (bound propagation via the existing back_end)
# ---------------------------------------------------------------------------


def _activation_layer_kinds(LayerKind) -> Tuple[str, ...]:
    """Layer kinds the state is defined over.

    ReLU splits at its kink; Sigmoid and Tanh split at their inflection point,
    where the same ``(lb < 0) & (ub > 0)`` test reads as "the box straddles the
    inflection" instead of "the box straddles the kink". Same test, weaker
    meaning: crossing it changes which way the neuron curves, not which affine
    piece it is in -- and measured on the ERAN sigmoid/tanh nets, interval
    propagation calls every deep-layer neuron unstable while only ~4% can
    actually be flipped, so this mask is a much looser superset there than it
    is on ReLU.
    """
    return (LayerKind.RELU.value, LayerKind.SIGMOID.value, LayerKind.TANH.value)


def _straddle(lb: torch.Tensor, ub: torch.Tensor, binning=None) -> torch.Tensor:
    """Which state coordinates the box can move across.

    Two-bin: the familiar `(lb < 0) & (ub > 0)`. Three-bin: one column per
    (neuron, wall) pair, `lb < wall < ub` -- and a neuron whose sign is pinned
    can still contribute a candidate through its +-tau walls, which is the
    whole point of the split on a smooth activation.
    """
    if binning is None:
        return (lb < 0) & (ub > 0)
    return binning.expand_bounds(lb, ub)


def _propagate_masks(
    wrapped_model: nn.Module,
    lb: torch.Tensor,
    ub: torch.Tensor,
    chunk: int = 25,
    reduce_any: bool = True,
    binning=None,
) -> Tuple[Optional[torch.Tensor], str]:
    """Interval-propagate the given boxes and return the unstable mask(s).

    reduce_any=True  -> [N], unstable for ANY of the boxes (the union).
    reduce_any=False -> [B, N], one row per box.

    Chunked because interval propagation holds bounds for every intermediate
    activation: a 100-lane pass over a ResNet is a large allocation, and the
    result is only booleans.
    """
    try:
        from act.back_end.analyze import analyze
        from act.back_end.core import Bounds, ConSet, Fact
        from act.back_end.layer_schema import LayerKind
        from act.back_end.verifier import find_entry_layer_id
        from act.pipeline.verification.torch2act import TorchToACT
    except Exception:
        return None, "back_end_import_failed"

    try:
        net = TorchToACT(wrapped_model).run()
        entry_id = find_entry_layer_id(net)
        out = None
        for start in range(0, lb.shape[0], max(1, chunk)):
            fact = Fact(
                bounds=Bounds(lb[start:start + chunk].clone(), ub[start:start + chunk].clone()),
                cons=ConSet(),
            )
            before, _after, _globalC = analyze(net, entry_id, fact)
            per_layer: List[torch.Tensor] = []
            for layer in net.layers:
                kind = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
                if kind not in _activation_layer_kinds(LayerKind):
                    continue
                bounds = before[layer.id].bounds
                unstable = _straddle(bounds.lb.flatten(start_dim=1),
                                     bounds.ub.flatten(start_dim=1), binning)
                per_layer.append(unstable.any(dim=0) if reduce_any else unstable)
            if not per_layer:
                return None, "no_activation_layers_in_net"
            part = torch.cat(per_layer, dim=-1)
            if out is None:
                out = part
            elif reduce_any:
                out = out | part
            else:
                out = torch.cat([out, part], dim=0)
        return out, "ok"
    except Exception as exc:
        # Pure-MLP graphs (mnist_fc and friends) come out of TorchToACT with
        # out_vars carrying the BATCH dimension: a Flatten inherits its
        # predecessor's variable count, and nothing normalises the batch away
        # the way _add_manual_conv2d does for convolutional graphs. Propagating
        # one row then trips "flatten out_vars length B*N != output elements N",
        # and the caller silently falls back to "every neuron is a candidate".
        #
        # Rather than change out_vars -- the coordinate system BaB, the verifier
        # and cons_exportor all share -- go along with it: hand the whole batch
        # in as ONE sample, which is the shape the graph was built for, then
        # fold the [1, B*N] result back to [B, N]. Retry only on that specific
        # failure; anything else is re-raised as before.
        if "flatten out_vars length" not in str(exc):
            return None, f"bound_propagation_failed: {type(exc).__name__}: {exc}"
        try:
            fact = Fact(bounds=Bounds(lb.reshape(1, -1).clone(),
                                      ub.reshape(1, -1).clone()), cons=ConSet())
            before, _after, _globalC = analyze(net, entry_id, fact)
            per_layer = []
            for layer in net.layers:
                kind = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
                if kind not in _activation_layer_kinds(LayerKind):
                    continue
                b = before[layer.id].bounds
                u = _straddle(b.lb.flatten().unsqueeze(0),
                              b.ub.flatten().unsqueeze(0), binning).squeeze(0)
                # [B*n*coords_per_neuron] for this layer -> [B, n*coords]
                per_layer.append(u.reshape(lb.shape[0], -1))
            if not per_layer:
                return None, "no_activation_layers_in_net"
            folded = torch.cat(per_layer, dim=1)          # [B, N]
            if reduce_any:
                return folded.any(dim=0), "ok_batched_graph"
            return folded, "ok_batched_graph"
        except Exception as exc2:
            return None, f"bound_propagation_failed: {type(exc2).__name__}: {exc2}"


def flip_balance_scores(
    wrapped_model: nn.Module,
    lb: torch.Tensor,
    ub: torch.Tensor,
    samples: int = 16,
) -> Optional[torch.Tensor]:
    """[B, N] float: how BALANCED each neuron's sign is under random points of
    its own box -- min(#positive, #negative) over `samples` draws.

    Interval propagation calls a neuron unstable if its bounds straddle zero,
    but that is an over-approximation: many such neurons still hold one sign
    for virtually every point actually in the box. As state coordinates those
    are dead weight -- they cost a dimension and carry no entropy, which is
    precisely what makes a high-dimensional pattern degenerate into
    "everything is novel". Balance is the empirical version of the question,
    and it needs no counterexample: any exploration of the box answers it.
    """
    from act.pipeline.fuzzing.mutations import _relu_sign_pattern_batched

    pos = None
    with torch.no_grad():
        for _ in range(max(1, samples)):
            x = lb + torch.rand_like(lb) * (ub - lb).clamp(min=0)
            sign = (_relu_sign_pattern_batched(wrapped_model, x) > 0)
            pos = sign.to(torch.int32) if pos is None else pos + sign.to(torch.int32)
    if pos is None:
        return None
    n = max(1, samples)
    return torch.minimum(pos, n - pos).to(torch.float32)


def ce_prior_from_registry(sm, total_neurons: int):
    """{instance: {"score": [N], "ce_sign": [N]}} over the FULL neuron axis.

    score_j = |P(sign_j = +1 | counterexamples) - P(sign_j = +1 | the rest)|,
    the continuous form of a spectrum-based suspiciousness, computed per
    instance from what this run recorded. Scattered back onto the full axis
    (the registry stores patterns restricted to the candidate set) so a later
    run can reuse it under a different mask.

    Only instances with BOTH populations get an entry -- a suspiciousness
    needs a failing and a passing side, and an instance with no counterexample
    has no failing side. That is exactly why this cannot bootstrap online, and
    why the transfer experiment feeds it in from a previous run instead.
    """
    cand = sm.candidate_indices.cpu()
    by_inst = {}
    for node in sm._registry:
        inst = int(node.payload["original_index"].reshape(-1)[0])
        b = by_inst.setdefault(inst, {"ce": [], "non": []})
        b["ce" if node.payload.get("is_ce") else "non"].append(node.point.cpu())

    out = {}
    for inst, b in by_inst.items():
        if not b["ce"] or not b["non"]:
            continue
        p_ce = (torch.stack(b["ce"]) > 0).float().mean(dim=0)
        p_non = (torch.stack(b["non"]) > 0).float().mean(dim=0)
        score = torch.zeros(total_neurons)
        sign = torch.ones(total_neurons)
        score[cand] = (p_ce - p_non).abs()
        sign[cand] = torch.where(p_ce >= 0.5, torch.ones_like(p_ce), -torch.ones_like(p_ce))
        out[inst] = {"score": score, "ce_sign": sign,
                     "n_ce": len(b["ce"]), "n_non": len(b["non"])}
    return out


def margin_gradient_scores(
    wrapped_model: nn.Module,
    output_spec,
    lb: torch.Tensor,
    ub: torch.Tensor,
    samples: int = 8,
) -> Optional[torch.Tensor]:
    """[B, N] float: mean |d severity / d z_j| over random points of each
    instance's own box, where severity is the property's own violation
    objective (what PGD ascends).

    Every other state-dimension criterion tried here ranks neurons by how much
    they MOVE -- balance under random sampling, either direction. Movement is a
    property of the geometry, not of the question being asked, and the arms
    built on it changed nothing: admission is supposed to say "is this state
    worth keeping", and a coordinate that moves a lot while being irrelevant to
    the property adds noise, not information.

    This ranks by how much a neuron's pre-activation matters to VIOLATING the
    property. Two samples differing only in low-gradient neurons then map to
    the same state -- which is correct, they are equivalent for the search --
    while differences in high-gradient neurons separate them. That coarsens
    the space (more collisions, so admission discriminates again) along the
    axis that carries the task.

    Sign is dropped: |grad| asks "does this neuron matter", not "which way".
    Which way is a question for the target, not for the coordinates.
    """
    from act.pipeline.fuzzing.mutations import _relu_preactivations_batched

    B = lb.shape[0]
    total = None
    for _ in range(max(1, samples)):
        preacts: List[torch.Tensor] = []
        handles = []

        def _hook(_module, inputs):
            if inputs and torch.is_tensor(inputs[0]):
                preacts.append(inputs[0])

        for module in wrapped_model.modules():
            if isinstance(module, nn.ReLU):
                handles.append(module.register_forward_pre_hook(_hook))
        try:
            x = lb + torch.rand_like(lb) * (ub - lb).clamp(min=0)
            out = wrapped_model(x)
        finally:
            for h in handles:
                h.remove()
        if not preacts:
            return None
        y = out["output"] if isinstance(out, dict) else out
        try:
            sev = output_spec.severity(y)
        except TypeError:
            sev = output_spec.severity(y, rows=torch.arange(B, device=y.device))
        except Exception:
            return None
        try:
            grads = torch.autograd.grad(sev.sum(), preacts, retain_graph=False,
                                        allow_unused=True)
        except Exception:
            return None
        parts = [
            (g if g is not None else torch.zeros_like(z)).flatten(start_dim=1).abs()
            for g, z in zip(grads, preacts)
        ]
        step = torch.cat(parts, dim=1).detach()
        total = step if total is None else total + step
    return None if total is None else total / max(1, samples)


def select_state_dims(
    submask: torch.Tensor,
    method: str,
    keep: float,
    scores: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Thin each instance's candidate row down to `keep` dimensions.

    submask: [I, C] bool, which candidates each instance may use.
    keep:    >=1 an absolute count, <1 a fraction of that instance's own row.
    scores:  [I, C] float, higher = keep first (required for non-random).

    Why thin at all: admission asks "has this state been seen", and the answer
    is only informative when the state space is coarse enough for the run to
    revisit it. An instance's own unstable set is ~883 dims on cifar100's
    medium ResNet -- 2^883 states against ~4000 samples a trial, so nothing
    ever repeats and the check accepts everything. The shipped row0 mask
    accidentally cut that to ~116 varying dims and did discriminate (20-27%
    rejected). This makes the cut deliberate, and lets the criterion be
    compared against the random baseline that isolates dimension count from
    which dimensions.
    """
    if keep <= 0:
        return submask
    out = torch.zeros_like(submask)
    for i in range(submask.shape[0]):
        idx = submask[i].nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        k = int(round(keep * idx.numel())) if keep < 1 else int(keep)
        k = max(1, min(k, idx.numel()))
        if method == "random" or scores is None or (
                method == "ce_prior" and float(scores[i].abs().sum()) == 0.0):
            # An instance with no prior (never cracked in the harvest run) has
            # an all-zero row; ranking it would just take the first k indices,
            # which is a systematic bias, not a fallback.
            pick = idx[torch.randperm(idx.numel(), device=idx.device)[:k]]
        elif method == "flip_rare":
            # Slowest-moving coordinates, but never the frozen ones. Admission
            # asks "seen this state before", so it only discriminates when the
            # coordinates COLLIDE -- and a coordinate collides when it rarely
            # changes. Measured: rejection tracks how much the chosen dims
            # move, monotonically (22.7% for the near-static row0 projection,
            # 9.9% random, 3.5% for the most-moving ones), so picking for
            # stillness is the deliberate version of what the row0 bug did by
            # accident. Balance 0 is excluded: a constant coordinate carries
            # nothing, and a state of all constants would make every sample
            # after the first look already-seen.
            row = scores[i][idx].to(idx.device)
            live = (row > 0).nonzero(as_tuple=True)[0]
            if live.numel() == 0:
                pick = idx[torch.randperm(idx.numel(), device=idx.device)[:k]]
            else:
                order = torch.argsort(row[live], descending=False)
                pick = idx[live[order[:k]]]
        else:
            order = torch.argsort(scores[i][idx].to(idx.device), descending=True)
            pick = idx[order[:k]]
        out[i, pick] = True
    return out


# Per-instance masks cost one interval-propagation pass EACH, and --repeat
# builds a fresh ACTFuzzer per trial while the instance boxes never change.
# Keyed on the box bytes so a rerun with the same specs is free.
_PER_INSTANCE_MASK_CACHE: Dict[bytes, torch.Tensor] = {}


def compute_gradient_budget_masks(
    wrapped_model: nn.Module,
    lb: torch.Tensor,
    ub: torch.Tensor,
    binning=None,
    threshold: float = 1.0,
) -> Tuple[Optional[torch.Tensor], str]:
    """[B, M] bool: which state coordinates the attack can actually afford.

    An alternative to `compute_unstable_mask`'s interval propagation, for the
    case where propagation has no resolution left. A coordinate is kept when
    the box's FIRST-ORDER budget for moving its pre-activation reaches the wall
    it would have to cross:

        eps . ||d z_j / d x||_1  >  |z_j(x0) - wall_m| * threshold

    Why this exists. Interval propagation's error MULTIPLIES across a product
    of two perturbed variables, which is exactly what self-attention is, so on
    a two-layer ViT it returns widths of 1e12 against a true reachable width of
    0.10 and calls 960/960 neurons unstable -- sound, and carrying no
    information. Measured on vit_2023 instance 0: interval 960/960, hybridz
    894/960, this criterion 74/960, computed in 0.9 s. On the ERAN sigmoid MLP
    it gives 24/600 where interval gives 355/600.

    This is NOT a sound bound -- it is a local first-order estimate, and both
    directions of error were measured on the sigmoid nets: of 24 neurons it
    called reachable, 20 really flipped, and 3 neurons flipped that it called
    unreachable. That is acceptable here and would not be in a verifier: the
    mask only CHOOSES ATTACK TARGETS. Missing a few costs a few candidates;
    soundness buys nothing when the sound answer is "all of them".

    Cost is one backward pass per neuron, independent of the lane count: lanes
    are independent in eval mode, so the gradient of ``z[:, j].sum()`` carries
    every lane's row for neuron j at once.
    """
    try:
        from act.pipeline.fuzzing.mutations import _relu_preactivations_batched
    except Exception:
        return None, "mutations_import_failed"

    try:
        x0 = ((lb + ub) / 2).detach()
        half = ((ub - lb) / 2).detach()
        xr = x0.clone().requires_grad_(True)
        z = _relu_preactivations_batched(wrapped_model, xr)
        B, N = z.shape
        budget = torch.zeros(B, N, device=z.device, dtype=z.dtype)
        for j in range(N):
            g, = torch.autograd.grad(z[:, j].sum(), xr, retain_graph=(j < N - 1))
            budget[:, j] = (g.detach().abs() * half).flatten(1).sum(1)
        z0 = z.detach()
    except Exception as exc:
        return None, f"gradient_budget_failed: {type(exc).__name__}: {exc}"

    if binning is None or binning.bins == 2:
        need = z0.abs()
        reach = budget > need * float(threshold)
        return reach, "ok"

    # Three-bin: one column per (neuron, wall), same interleaved order the code
    # uses, so the mask indexes the coordinate space and not the neuron space.
    walls = binning._walls(N, z0)
    need = (z0.repeat_interleave(2, dim=-1) - walls).abs()
    budget2 = budget.repeat_interleave(2, dim=-1)
    return budget2 > need * float(threshold), "ok"


def compute_per_instance_masks(
    wrapped_model: nn.Module,
    lb: torch.Tensor,
    ub: torch.Tensor,
    binning=None,
) -> Tuple[Optional[torch.Tensor], str]:
    """[B, N] bool: row b is instance b's OWN unstable set.

    The union of these rows is what scope="union" produces, and every row of it
    is what scope="row0" wrongly substitutes with row 0's. Keeping the rows
    separate is what lets a lane be steered inside its own set instead of the
    batch's: measured, an instance's set is ~5% of the union, so choosing flip
    targets from the union at random misses the lane's own neurons 95% of the
    time, and novelty measured over the union is diluted to always-novel.
    """
    key = (lb.detach().cpu().numpy().tobytes() + b"|"
           + ub.detach().cpu().numpy().tobytes() + b"|"
           + str(getattr(binning, "bins", 2)).encode()
           + str(getattr(binning, "tau", 0.0)).encode())
    cached = _PER_INSTANCE_MASK_CACHE.get(key)
    if cached is not None:
        return cached.to(lb.device), "cached"

    # One propagation per CHUNK, not per instance. Interval propagation is
    # per-sample independent -- bounds.lb comes back [batch, neurons], row i
    # being row i's box -- so a chunk of 25 boxes yields 25 masks from one
    # pass. The obvious loop (broadcast instance i's box to B lanes, keep lane
    # 0) computes B identical lanes and throws B-1 away, which is 100x the
    # work for the same answer.
    out, reason = _propagate_masks(wrapped_model, lb, ub, chunk=25,
                                   reduce_any=False, binning=binning)
    if out is None:
        return None, reason
    _PER_INSTANCE_MASK_CACHE[key] = out
    return out.to(lb.device), "ok"


def compute_unstable_mask(
    wrapped_model: nn.Module,
    lb: torch.Tensor,
    ub: torch.Tensor,
    scope: str = "row0",
    chunk: int = 25,
    binning=None,
) -> Tuple[Optional[torch.Tensor], str]:
    """Boolean mask [M] over every state COORDINATE, True where the
    neuron is UNSTABLE (pre-activation interval straddles zero for some
    input in the box [lb, ub]), in the same flattened order as
    act.pipeline.fuzzing.mutations._relu_preactivations_batched.

    Computed via interval bound propagation through the existing
    act.back_end verifier (TorchToACT + analyze()) -- the same
    ``(lb < 0) & (ub > 0)`` test used by the BaB branching code
    (act/back_end/solver/solver_dual.py) and (act/back_end/bab/branching).

    Args:
        wrapped_model: The synthesized model, output spec included. Its spec
            rows are row-indexed by lane, so any forward pass through it must
            present exactly as many lanes as the spec has rows.
        lb, ub: The FULL batch box, ``[B, ...]``, one row per synthesized spec
            row -- not a single lane. Bound propagation itself runs on lane 0
            only (the mask is a per-group approximation, see below), but the
            shape-check forward needs all B lanes or row-indexed specs such as
            TOP1_ROBUST reject it.

    Returns:
        ``(mask, reason)``. ``mask`` is None -- meaning "treat every neuron as
        a candidate" -- if bound propagation fails for any reason, or if the
        per-layer neuron count derived from the Net graph doesn't match a
        forward-hook pass over the same model (a correctness guard against
        silently relying on the act.back_end Net's layer ordering matching
        nn.Module.modules() order for architectures where that isn't
        guaranteed). ``reason`` is "ok" on success and otherwise names which
        of those bailed, so the caller can report a dropped mask instead of
        degrading silently -- this used to be swallowed by a bare
        ``except Exception``, which hid a shape bug for the whole
        TOP1_ROBUST x state-admission combination.

    Note:
        The mask describes lane 0's box and is then applied to the whole
        group. That is a pre-existing approximation, not something this
        function guarantees: each row of a batched group carries its own box,
        so a neuron unstable for instance 0 may be stable for instance 7.
    """
    try:
        from act.back_end.analyze import analyze
        from act.back_end.core import Bounds, ConSet, Fact
        from act.back_end.layer_schema import LayerKind
        from act.back_end.verifier import find_entry_layer_id
        from act.pipeline.verification.torch2act import TorchToACT
    except Exception:
        return None, "back_end_import_failed"

    # scope="row0" propagates ONLY the first spec row's box and keeps only
    # lane 0's mask, then applies it to every lane. That is what shipped, and
    # it is wrong for a batch: measured on cifar100_2024, an instance's own
    # unstable set overlaps lane 0's by 12.2% (medium) / 9.1% (large), so
    # 88-91% of the neurons a given lane can actually flip are invisible to
    # everything scoped by this mask -- HPGD's flip candidates,
    # PatternStateManager's novelty check, the sparsity weights, the frontier.
    # Only 6.6-7.7% of the sign flips that actually occur land inside it.
    #
    # scope="union" propagates every row's box and ORs the masks, so the
    # result is a true superset of each lane's own unstable set. Bigger (and
    # so more expensive downstream), but it can no longer ask a lane to flip a
    # neuron its own box holds fixed. Kept opt-in: every arm recorded before
    # 2026-08-27 ran "row0", and silently changing the default would make
    # those numbers unreproducible.
    try:
        # Always propagate the FULL batch and reduce afterwards, even for
        # "row0". Slicing to lb[:1] first would hide the batch from the
        # batched-graph fallback in _propagate_masks, which needs every row to
        # reconstruct the per-lane masks; row0 then just takes row 0 of the
        # result, which is the same mask it computed before.
        rows_mask, reason = _propagate_masks(wrapped_model, lb, ub, chunk,
                                             reduce_any=False, binning=binning)
        if rows_mask is None:
            return None, reason
        mask = rows_mask.any(dim=0) if scope == "union" else rows_mask[0]
    except Exception as exc:
        return None, f"bound_propagation_failed: {type(exc).__name__}: {exc}"

    try:
        from act.pipeline.fuzzing.mutations import _relu_preactivations_batched

        with torch.no_grad():
            # All B lanes: the wrapped model evaluates its output spec on
            # every forward, and a row-indexed spec refuses a lane count that
            # doesn't match its row count.
            sample = lb + torch.rand_like(lb) * (ub - lb).clamp(min=0)
            z = _relu_preactivations_batched(wrapped_model, sample)
        # The mask is per COORDINATE, the forward pass per neuron; under a
        # three-bin state those differ by the coordinates-per-neuron factor.
        per_neuron = 1 if binning is None else binning.num_coords // binning.num_neurons
        if z.shape[1] * per_neuron != mask.shape[0]:
            return None, (
                f"neuron_count_mismatch: net graph says {mask.shape[0]} coords, "
                f"forward hooks say {z.shape[1]} neurons x {per_neuron}"
            )
    except Exception as exc:
        return None, f"shape_check_forward_failed: {type(exc).__name__}: {exc}"

    return mask.to(device=lb.device), "ok"


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
        # Cached [R, C] view of _registry for propose_targets, rebuilt whenever
        # the registry grows. Kept here rather than recomputed per call: the
        # registry only ever appends.
        self._registry_points: Optional[torch.Tensor] = None
        self._registry_inst: Optional[torch.Tensor] = None
        # Swap in any [M, C] -> [M] callable to steer target proposal by a
        # state evaluation of your own; None uses rarity_score.
        self.target_score_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
        # Per-instance exploration frontier, for the coarse-to-fine schedule.
        # Radius is measured from the instance's FIRST admitted pattern (its
        # anchor) rather than as a true diameter: a diameter needs all pairs,
        # this needs one Hamming distance per admission and answers the same
        # question -- is this instance's discovered region still growing.
        # [I, C] bool over the candidate axis: which candidates each instance
        # can actually flip. None = every lane may use every candidate (the
        # original behaviour).
        self.instance_submask: Optional[torch.Tensor] = None
        self._inst_anchor: Dict[int, torch.Tensor] = {}
        self._inst_radius: Dict[int, int] = {}
        self._inst_stall: Dict[int, int] = {}
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

    def scope_to_instance(
        self, restricted: torch.Tensor, original_indices: torch.Tensor
    ) -> torch.Tensor:
        """Pin candidate positions this lane's own box cannot flip to +1.

        Without this, a union-scoped candidate axis makes novelty trivially
        true: the axis carries every instance's unstable neurons, so a lane's
        pattern differs from every other lane's at positions neither of them
        can control, and admission degenerates to accepting everything
        (measured: already-seen rate fell to 0.0% under the union mask).
        Pinning them to a constant leaves only the positions the lane can
        actually move, which is what the fingerprint should be keyed on."""
        if self.instance_submask is None:
            return restricted
        idx = original_indices.reshape(-1).to(self.instance_submask.device)
        sub = self.instance_submask[idx].to(restricted.device)
        return torch.where(sub, restricted, torch.ones_like(restricted))

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
        keys = self.fingerprint(self.scope_to_instance(
            self.restrict(patterns_full.reshape(B, -1)), original_indices))
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
        restricted = self.scope_to_instance(
            self.restrict(patterns_full.reshape(B, -1)), original_indices)
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
            self._note_frontier(int(original_indices[b]), restricted[b])
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
        self._note_frontier(int(payload["original_index"].reshape(-1)[0]), pattern)
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

    # -- exploration frontier (coarse-to-fine schedule) ------------------------

    def _note_frontier(self, inst: int, restricted: torch.Tensor) -> None:
        """Fold one admitted pattern into its instance's frontier radius."""
        anchor = self._inst_anchor.get(inst)
        if anchor is None:
            self._inst_anchor[inst] = restricted.detach().clone()
            self._inst_radius[inst] = 0
            self._inst_stall[inst] = 0
            return
        d = int((anchor != restricted).sum())
        if d > self._inst_radius.get(inst, 0):
            self._inst_radius[inst] = d
            self._inst_stall[inst] = 0
        else:
            self._inst_stall[inst] = self._inst_stall.get(inst, 0) + 1

    def frontier(self, inst: int) -> Tuple[int, int]:
        """(radius, admissions since the radius last grew) for one instance."""
        return self._inst_radius.get(inst, 0), self._inst_stall.get(inst, 0)

    def refine_mask(self, original_indices: torch.Tensor, patience: int) -> torch.Tensor:
        """[B] bool: which lanes should switch from expanding the frontier to
        navigating inside it.

        An instance stays in the COARSE phase while its radius is still
        growing -- large random flips are imprecise but displace far, which is
        what pushes the boundary out and stocks the registry with the distant
        real states interpolation needs as endpoints. Once `patience`
        consecutive admissions fail to grow the radius, that instance has
        stopped expanding and its lanes switch to the FINE phase. The test is
        per instance, so one benchmark can have lanes in both phases at once,
        and an instance whose radius starts growing again drops back to coarse
        on its own."""
        return torch.tensor(
            [self._inst_stall.get(int(i), 0) >= patience for i in original_indices.tolist()],
            dtype=torch.bool,
        )

    # -- target proposal ------------------------------------------------------

    def rarity_score(self, candidates_restricted: torch.Tensor) -> torch.Tensor:
        """[M] score for M proposed patterns: mean rarity of the signs they
        name, from the same marginal occupancy `sparsity_weights` uses. This is
        the DEFAULT scorer for propose_targets, and it is meant to be replaced:
        set `target_score_fn` to any callable [M, C] -> [M] (higher = aim here)
        to steer at whatever a later state evaluation decides is worth
        reaching. Uniform until something has been admitted."""
        M = candidates_restricted.shape[0]
        if self._sign_total == 0:
            return torch.ones(M, device=candidates_restricted.device)
        freq_plus = (self._sign_count / self._sign_total).to(
            candidates_restricted.device, torch.float32
        )
        # Probability of the sign each candidate names, per neuron; rare = high.
        p = torch.where(candidates_restricted > 0, freq_plus, 1.0 - freq_plus)
        return (1.0 - p).mean(dim=1)

    def propose_targets(
        self,
        natural_restricted: torch.Tensor,
        original_indices: torch.Tensor,
        rhos: Sequence[float] = (0.25, 0.5, 0.75),
        pool: int = 32,
        proposals: int = 2,
        min_d: int = 1,
        rho_per_lane: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """[B, C] target patterns that are plausibly FEASIBLE and UNVISITED.

        Flipping k bits independently, which is what HPGDMutation does by
        default, names a sign assignment nothing guarantees is jointly
        satisfiable -- measured, it is reached ~15% of the time and exactly
        ~0%. A pattern from this registry is feasible but already admitted, so
        arriving at it exactly buys no novelty; and one from ANOTHER instance
        is not feasible here at all (same weights, different input box), which
        measures the same ~15% as a random flip.

        What is left is the space BETWEEN two real states of the SAME instance:
        take a registry pattern P for this lane, and flip only a rho-share of
        the positions where P differs from the seed. Those positions are
        jointly realizable transitions -- both endpoints are real -- so the
        named cell has a structural reason to be non-empty that random flips
        lack, while not itself having been visited. Measured: reached exactly
        ~95-97% of the time, with the target unvisited 100% of the time.

        Candidates are scored by `target_score_fn` (default `rarity_score`) and
        the best one per lane is returned; that scorer is the seam for steering
        toward whatever states are judged worth reaching.

        Returns (targets [B, C], proposed [B] bool). A lane with no
        same-instance neighbour at distance >= `min_d`, or whose every
        candidate has already been visited, is left at its own natural pattern
        and marked False -- the caller MUST fall back to random flips for it,
        or that lane asks for nothing and its whole projection step is wasted.

        `pool` bounds the cost: the registry is scanned per lane, so it is
        SUBSAMPLED to `pool` same-instance entries rather than walked whole.
        `rho_per_lane`, when given, overrides `rhos` for that lane -- which is
        how the fine phase anneals from coarse interpolation toward precise.
        """
        B, C = natural_restricted.shape
        proposed = torch.zeros(B, dtype=torch.bool)
        if not self._registry:
            return natural_restricted.clone(), proposed
        device = natural_restricted.device
        if self._registry_points is None or self._registry_points.shape[0] != len(self._registry):
            self._registry_points = torch.stack([n.point for n in self._registry]).to(device)
            self._registry_inst = torch.tensor(
                [int(n.payload["original_index"].item()) for n in self._registry],
                device=device,
            )
        points, inst = self._registry_points, self._registry_inst

        out = natural_restricted.clone()
        score_fn = self.target_score_fn or self.rarity_score
        for b in range(B):
            same = (inst == int(original_indices[b])).nonzero(as_tuple=True)[0]
            if same.numel() == 0:
                continue
            if same.numel() > pool:
                same = same[torch.randperm(same.numel(), device=device)[:pool]]
            nat = natural_restricted[b]
            d = (points[same] != nat).sum(dim=1)
            same = same[(d >= max(1, min_d)).nonzero(as_tuple=True)[0]]
            if same.numel() == 0:
                continue

            lane_rhos = (float(rho_per_lane[b]),) if rho_per_lane is not None else rhos
            cands = []
            for _ in range(proposals):
                j = same[torch.randint(same.numel(), (1,), device=device)]
                diff = (points[j].reshape(-1) != nat).nonzero(as_tuple=True)[0]
                if diff.numel() == 0:
                    continue
                for rho in lane_rhos:
                    # Never the whole difference set: that IS the neighbour, an
                    # already-visited state, so it would be filtered out below
                    # anyway -- capping keeps a candidate slot from being wasted.
                    take = min(int(round(float(rho) * diff.numel())), max(1, diff.numel() - 1))
                    take = max(1, take)
                    pick = diff[torch.randperm(diff.numel(), device=device)[:take]]
                    c = nat.clone()
                    c[pick] *= -1
                    cands.append(c)
            if not cands:
                continue
            cand = torch.stack(cands)                                  # [M, C]

            # Drop anything already recorded for this lane -- the whole point
            # is to name a state admission has not seen.
            keys = self.fingerprint(cand)
            idx_col = torch.full((cand.shape[0], 1), int(original_indices[b]),
                                 device=keys.device, dtype=keys.dtype)
            keys_cpu = torch.cat([idx_col, keys], dim=1).cpu().numpy()
            unseen = [m for m in range(cand.shape[0])
                      if keys_cpu[m].tobytes() not in self._seen]
            if not unseen:
                continue
            cand = cand[unseen]

            out[b] = cand[int(torch.argmax(score_fn(cand)))]
            proposed[b] = True

        return out, proposed

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
