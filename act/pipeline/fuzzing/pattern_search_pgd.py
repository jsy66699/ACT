"""
PatternSearchPGD: pattern-space PGD attack with a collect/attack split, run
over up to three rounds.

Round 1 (broad probe): repeatedly sample a fresh random point in the
instance's input box, build a target ReLU sign pattern by flipping a random
subset of that point's own natural pattern, and run a margin-loss
projected-gradient walk toward that target. Every projected point is kept in
a diversity pool only if its actually-achieved sign pattern differs from
every pattern already pooled by at least --min-diversity-hamming neurons, so
the pool ends up covering many different activation regions instead of
near-duplicates of the same one. Every collection round is immediately
followed by an attack phase: real violation-maximizing PGD from every newly
pooled point, checked against the instance's OutputSpec for an actual
counterexample.

Round 2 (empirically-focused probe, opt-in via --round2-iterations): round
1's pool empirically reveals which neurons actually flipped at least once
(as opposed to the bound-propagation notion of "unstable" -- this is a
strictly-observed, possibly tighter subset). Round 2 restricts flip
candidates to just that subset and repeats round 1's collect+attack cycle.

Round 3 (anchor pull-back, opt-in via --round3-iterations, only runs if
round 2 found a real counterexample): each attempt starts from a random
round-2 counterexample's own (input, achieved pattern) as an anchor, nudges
the anchor input by a small random perturbation (--round3-noise-scale;
starting exactly at the anchor gives zero margin-loss gradient), then
gradient-walks back toward the anchor's OWN pattern -- stopping as soon as
the achieved Hamming distance to it is <= --round3-hamming-radius. This
densely samples new inputs near a known-good (i.e. actually violating)
region instead of exploring blindly, exploiting round 2's find.

None of the three rounds hill-climbs toward any single target within a
round: every collection attempt starts from an independent random point (or
anchor, in round 3) and a freshly randomized target pattern, trading
exploitation for breadth of pattern coverage before spending attack-PGD
budget.

Usage:
    python -m act.pipeline.fuzzing.pattern_search_pgd \\
        --category mnist_fc --collect-iterations 200 --flip-count 10 \\
        --round2-iterations 200 --round3-iterations 200

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import secrets
import statistics
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from act.front_end.model_synthesis import synthesize_models_from_specs
from act.front_end.specs import InKind, OutKind, OutputSpec
from act.front_end.verifiable_model import InputSpecLayer, OutputSpecLayer
from act.front_end.vnnlib_loader.create_specs import VNNLibSpecCreator
from act.pipeline.fuzzing.checker import Counterexample, PropertyChecker
from act.pipeline.fuzzing.corpus import FuzzingSeed
from act.util.cli_utils import add_device_args, initialize_from_args
from act.util.path_config import get_pipeline_log_dir

DEFAULT_OUTPUT_DIR = Path(get_pipeline_log_dir()) / "pattern_search_pgd"


def _extract_spec(model: nn.Module, layer_type: type) -> Any | None:
    for layer in model.children():
        if isinstance(layer, layer_type):
            return layer.spec
    return None


def _relu_sign_pattern(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Concrete ReLU pre-activation sign pattern (+-1) at x, flattened and
    concatenated across every ReLU layer in module-registration order."""
    preacts: list[torch.Tensor] = []
    handles = []

    def hook(_module, inputs):
        if inputs and torch.is_tensor(inputs[0]):
            preacts.append(inputs[0].detach())

    for module in model.modules():
        if isinstance(module, nn.ReLU):
            handles.append(module.register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for handle in handles:
            handle.remove()
    if not preacts:
        raise RuntimeError("Model has no ReLU layers reachable from a forward pass on this input.")
    z_all = torch.cat([z.flatten(start_dim=1) for z in preacts], dim=1)
    return torch.where(z_all > 0, torch.ones_like(z_all), -torch.ones_like(z_all))[0]


def _relu_preactivations(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Same as `_relu_sign_pattern` but keeps the autograd graph and returns
    the raw pre-activation values (used for the margin-loss gradient)."""
    preacts: list[torch.Tensor] = []
    handles = []

    def hook(_module, inputs):
        if inputs and torch.is_tensor(inputs[0]):
            preacts.append(inputs[0])

    for module in model.modules():
        if isinstance(module, nn.ReLU):
            handles.append(module.register_forward_pre_hook(hook))
    try:
        model(x)
    finally:
        for handle in handles:
            handle.remove()
    if not preacts:
        raise RuntimeError("Model has no ReLU layers reachable from a forward pass on this input.")
    return torch.cat([z.flatten(start_dim=1) for z in preacts], dim=1)


def _hamming(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((a != b).sum().item())


class _PatternBKTree:
    """BK-tree over Hamming distance for ReLU sign-pattern deduplication.

    Phase 1's diversity filter needs to answer "is any pooled point within
    min_diversity_hamming of this candidate?" for every attempt. A linear
    scan over the whole pool is O(n) per attempt, which gets slow once the
    pool grows into the thousands. A BK-tree answers the same threshold
    query in roughly O(log n) average case by pruning subtrees via the
    triangle inequality.
    """

    def __init__(self) -> None:
        self._root: tuple[torch.Tensor, dict[int, Any]] | None = None
        self.size = 0

    def has_within(self, point: torch.Tensor, threshold: int) -> bool:
        if self._root is None:
            return False
        stack = [self._root]
        while stack:
            node_point, children = stack.pop()
            d = _hamming(node_point, point)
            if d <= threshold:
                return True
            for dist, child in children.items():
                if abs(dist - d) <= threshold:
                    stack.append(child)
        return False

    def insert(self, point: torch.Tensor) -> None:
        if self._root is None:
            self._root = (point, {})
            self.size += 1
            return
        node = self._root
        while True:
            node_point, children = node
            d = _hamming(node_point, point)
            if d == 0:
                return
            if d in children:
                node = children[d]
            else:
                children[d] = (point, {})
                self.size += 1
                return


def _flip_pattern(pattern: torch.Tensor, flip_count: int, candidates: torch.Tensor) -> torch.Tensor:
    mutated = pattern.clone()
    k = min(int(flip_count), int(candidates.numel()))
    if k <= 0:
        return mutated
    perm = torch.randperm(candidates.numel(), device=pattern.device)[:k]
    mutated[candidates[perm]] *= -1
    return mutated


def _project_to_pattern(
    x0: torch.Tensor,
    model: nn.Module,
    lb: torch.Tensor,
    ub: torch.Tensor,
    target_signs: torch.Tensor,
    steps: int,
    margin: float,
    step_size: Optional[float],
) -> torch.Tensor:
    """Sign-gradient descent toward a target ReLU sign pattern: minimizes a
    hinge loss that penalizes any neuron whose pre-activation sign disagrees
    with `target_signs` by less than `margin`, clamped to [lb, ub] every
    step. Stops early once every neuron already clears the margin."""
    if step_size is None:
        step_size = float((ub - lb).abs().max().item()) / max(steps, 1)
        step_size = max(step_size, 1e-6)
    x = x0.detach().clone()
    for _ in range(steps):
        x_req = x.detach().clone().requires_grad_(True)
        z_all = _relu_preactivations(model, x_req)
        violation = (margin - target_signs.unsqueeze(0) * z_all).clamp(min=0)
        loss = violation.sum()
        if float(loss.item()) == 0.0:
            break
        grad = torch.autograd.grad(loss, x_req)[0]
        x = x_req.detach() - step_size * torch.sign(grad)
        x = torch.max(torch.min(x, ub), lb)
    return x.detach()


def _violation_score(
    output_spec: Optional[OutputSpec], outputs: torch.Tensor, label: torch.Tensor
) -> torch.Tensor:
    """Differentiable per-sample score; higher = closer to (or past)
    violating the property, used as the phase-2 PGD attack loss. Mirrors
    PropertyChecker's own sign conventions for the OutKinds it supports
    (LINEAR_LE, TOP1_ROBUST, MARGIN_ROBUST). Other kinds (e.g. RANGE,
    UNSAFE_LINEAR) fall back to output-variance maximization, matching
    act.pipeline.fuzzing.mutations.PGDMutation's unsupervised fallback."""
    if output_spec is None:
        return outputs.var(dim=1)
    if output_spec.kind == OutKind.LINEAR_LE and output_spec.c is not None and output_spec.d is not None:
        c = output_spec.c.to(outputs.device)
        d = float(output_spec.d)
        return (outputs * c).sum(dim=1) - d
    if output_spec.kind in (OutKind.TOP1_ROBUST, OutKind.MARGIN_ROBUST):
        y = label.clamp(min=0).to(outputs.device).view(-1, 1)
        target_logit = outputs.gather(1, y).squeeze(1)
        other = outputs.scatter(1, y, float("-inf"))
        margin = other.max(dim=1).values - target_logit
        if output_spec.kind == OutKind.MARGIN_ROBUST:
            threshold = output_spec.margin
            threshold = float(threshold.item()) if torch.is_tensor(threshold) else float(threshold or 0.0)
            margin = margin + threshold
        return margin
    return outputs.var(dim=1)


def _attack_pgd(
    x0: torch.Tensor,
    model: nn.Module,
    lb: torch.Tensor,
    ub: torch.Tensor,
    output_spec: Optional[OutputSpec],
    label: torch.Tensor,
    steps: int,
    step_size: Optional[float],
) -> torch.Tensor:
    """Real violation-maximizing PGD: ascends `_violation_score`, projected
    to [lb, ub] every step."""
    if step_size is None:
        step_size = float((ub - lb).abs().max().item()) / max(steps, 1)
        step_size = max(step_size, 1e-6)
    x = x0.detach().clone()
    for _ in range(steps):
        x_req = x.detach().clone().requires_grad_(True)
        out = model(x_req)
        if isinstance(out, dict):
            out = out["output"]
        loss = _violation_score(output_spec, out, label).sum()
        grad = torch.autograd.grad(loss, x_req)[0]
        x = x_req.detach() + step_size * torch.sign(grad)
        x = torch.max(torch.min(x, ub), lb)
    return x.detach()


def run_pattern_search_pgd(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_instances = args.max_instances
    if args.instance_index is not None:
        load_instances = max(load_instances, args.instance_index + 1)

    print(f"[PatternSearchPGD] Loading VNNLIB category={args.category!r}")
    spec_results = VNNLibSpecCreator().create_specs_for_data_model_pairs(
        categories=[args.category], max_instances=load_instances,
    )
    if not spec_results:
        from act.front_end.vnnlib_loader.data_model_loader import list_local_categories

        if args.category in list_local_categories():
            raise RuntimeError(
                f"category={args.category!r} is downloaded, but every instance failed to parse into a spec "
                "(see the 'Failed to create specs for ...' warnings above -- a common cause is that the "
                "on-disk .vnnlib files are VNNLIB 1.0 (flat) format, which this ACT build no longer accepts; "
                "it requires VNNLIB 2.0 files declaring (vnnlib-version)/(declare-network))."
            )
        raise RuntimeError(
            f"No VNNLIB specs were loaded for category={args.category!r}: this category is not downloaded. "
            f"Download it first with: python -m act.pipeline --download {args.category}"
        )
    if args.instance_index is not None:
        if not (0 <= args.instance_index < len(spec_results)):
            raise ValueError(
                f"--instance-index {args.instance_index} is out of range for "
                f"{len(spec_results)} loaded instance(s)."
            )
        spec_results = [spec_results[args.instance_index]]
    else:
        spec_results = spec_results[:1]

    spec_result = spec_results[0]
    _category, instance_id, _model, labeled_tensors, _spec_pairs = spec_result

    print(f"[PatternSearchPGD] Synthesizing wrapped model(s) for instance={instance_id!r}")
    wrapped_models = synthesize_models_from_specs([spec_result])
    if not wrapped_models:
        raise RuntimeError("ACT model synthesis produced no wrapped models.")

    model_items = list(wrapped_models.items())
    if not (0 <= args.model_index < len(model_items)):
        raise ValueError(
            f"--model-index {args.model_index} is out of range for "
            f"{len(model_items)} synthesized wrapped model(s)."
        )
    model_id, wrapped_model = model_items[args.model_index]
    wrapped_model.eval()

    device = next(wrapped_model.parameters()).device
    input_spec = _extract_spec(wrapped_model, InputSpecLayer)
    output_spec = _extract_spec(wrapped_model, OutputSpecLayer)
    if input_spec is None or input_spec.kind not in (InKind.BOX, InKind.LINF_BALL, InKind.LP_EMBEDDING):
        raise ValueError(
            "PatternSearchPGD requires a BOX, LINF_BALL, or LP_EMBEDDING InputSpec, got "
            f"{None if input_spec is None else input_spec.kind!r}."
        )

    lb, ub = input_spec.materialize_box_seed()
    lb = lb.to(device)[:1]
    ub = ub.to(device)[:1]

    labeled = labeled_tensors[0]
    if labeled.label is not None:
        label = labeled.label.to(device=device, dtype=torch.long).view(-1)
    else:
        label = torch.full((1,), -1, dtype=torch.long, device=device)

    checker = PropertyChecker(output_spec)

    def sample_point() -> torch.Tensor:
        return (lb + torch.rand_like(lb) * (ub - lb).clamp(min=0)).detach()

    total_relu_neurons = int(_relu_sign_pattern(wrapped_model, sample_point()).numel())
    print(f"[PatternSearchPGD] model={model_id}, total_relu_neurons={total_relu_neurons}")
    all_indices = torch.arange(total_relu_neurons, device=device)

    # ---------------- shared collect/attack state across all rounds ----------------
    pool_points: list[torch.Tensor] = []
    pool_patterns: list[torch.Tensor] = []
    tree = _PatternBKTree()
    # threshold for has_within is "closer than min_diversity_hamming", i.e.
    # reject if some pooled point is within (min_diversity_hamming - 1).
    reject_threshold = max(0, args.min_diversity_hamming - 1)
    collect_rows: list[dict[str, Any]] = []
    all_attack_rows: list[dict[str, Any]] = []
    round_attack_stats: dict[str, dict[str, Any]] = {}
    all_counterexamples: list[Counterexample] = []
    saved_ce_count = 0

    def collect_round(round_name: str, iterations: int, timeout: float, candidates: torch.Tensor) -> int:
        """Round 1/2: independent random-flip projection attempts against
        `candidates`, appending diverse results into the shared pool/tree.
        Returns the number of attempts actually made."""
        round_start = time.time()
        made = 0
        while made < iterations:
            if time.time() - round_start >= timeout:
                print(f"[PatternSearchPGD] {round_name} timeout reached after {made} attempts.")
                break
            if args.pool_target > 0 and len(pool_points) >= args.pool_target:
                print(f"[PatternSearchPGD] {round_name}: pool reached target size {args.pool_target}, stopping early.")
                break
            made += 1

            base_point = sample_point()
            natural_pattern = _relu_sign_pattern(wrapped_model, base_point)
            target_pattern = _flip_pattern(natural_pattern, args.flip_count, candidates)
            flip_retry_count = 0
            while flip_retry_count < args.flip_retry_attempts and tree.has_within(target_pattern, reject_threshold):
                target_pattern = _flip_pattern(natural_pattern, args.flip_count, candidates)
                flip_retry_count += 1

            projected = _project_to_pattern(
                base_point, wrapped_model, lb, ub, target_pattern,
                args.pattern_steps, args.pattern_margin, args.pattern_step_size,
            )
            achieved_pattern = _relu_sign_pattern(wrapped_model, projected)
            kept = not tree.has_within(achieved_pattern, reject_threshold)
            if kept:
                pool_points.append(projected)
                pool_patterns.append(achieved_pattern)
                tree.insert(achieved_pattern)

            collect_rows.append({
                "round": round_name, "attempt": made, "kept": bool(kept),
                "pool_size_after": len(pool_points), "flip_retry_count": flip_retry_count,
            })
            if args.report_interval > 0 and made % args.report_interval == 0:
                print(f"[PatternSearchPGD] {round_name} attempts={made}/{iterations}, pool_size={len(pool_points)}")
        return made

    def attack_phase(points: list[torch.Tensor], round_name: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Real violation-maximizing PGD on `points`, checked for actual
        counterexamples. Returns (final adversarial input, achieved ReLU
        sign pattern) for every point that turned out to be a real
        counterexample -- used by round 3 as anchors."""
        nonlocal saved_ce_count
        print(f"[PatternSearchPGD] Attacking {len(points)} pooled point(s) from {round_name}...")
        attack_start = time.time()
        round_ce_count = 0
        ce_records: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, point in enumerate(points):
            adv_input = _attack_pgd(
                point, wrapped_model, lb, ub, output_spec, label, args.pgd_steps, args.pgd_step_size,
            )
            with torch.no_grad():
                out = wrapped_model(adv_input)
                outputs = out["output"] if isinstance(out, dict) else out
            seeds = FuzzingSeed(
                tensor=adv_input, original_tensor=point,
                original_index=torch.zeros(1, dtype=torch.long, device=device), label=label,
            )
            violation_mask, batch_ces = checker.check(inputs=adv_input, outputs=outputs, seeds=seeds)
            is_ce = bool(violation_mask[0].item())
            score = float(_violation_score(output_spec, outputs, label)[0].item())

            all_attack_rows.append({
                "round": round_name, "pool_index": i, "is_counterexample": is_ce, "violation_score": score,
            })
            if is_ce:
                round_ce_count += 1
                # Use the FINAL adversarial input's own pattern, not the pre-attack-PGD
                # projection's pattern -- the attack PGD steps can shift it.
                ce_records.append((adv_input.detach(), _relu_sign_pattern(wrapped_model, adv_input)))
                for ce in batch_ces:
                    all_counterexamples.append(ce)
                    if not args.no_save:
                        ce.save(output_dir / f"ce_{saved_ce_count + 1}.pt")
                    saved_ce_count += 1

            if args.report_interval > 0 and (i + 1) % args.report_interval == 0:
                print(f"[PatternSearchPGD] {round_name} attack {i + 1}/{len(points)}, counterexamples={round_ce_count}")

        elapsed = time.time() - attack_start
        rate = 100.0 * round_ce_count / len(points) if points else 0.0
        print(
            f"[PatternSearchPGD] {round_name} attack done: {round_ce_count}/{len(points)} counterexamples "
            f"({rate:.1f}%) in {elapsed:.1f}s."
        )
        round_attack_stats[round_name] = {
            "pool_size": len(points), "counterexamples_found": round_ce_count,
            "counterexample_rate": rate, "time_seconds": float(elapsed),
        }
        return ce_records

    # ---------------- Round 1: broad probe over all candidate neurons ----------------
    phase1_start = time.time()
    print(
        f"[PatternSearchPGD] Round 1: broad probe over {all_indices.numel()} candidate neurons, "
        f"{args.collect_iterations} attempts (min pairwise Hamming distance {args.min_diversity_hamming})..."
    )
    round1_attempts = collect_round("round1", args.collect_iterations, args.collect_timeout, all_indices)
    round1_pool_size = len(pool_points)
    print(f"[PatternSearchPGD] Round 1 done: {round1_pool_size} points kept out of {round1_attempts} attempts.")
    attack_phase(pool_points[:round1_pool_size], "round1")

    # ---------------- Round 2: empirically-focused probe (opt-in) ----------------
    round2_attempts = 0
    round3_attempts = 0
    round2_pool_size = round1_pool_size
    round2_ce_records: list[tuple[torch.Tensor, torch.Tensor]] = []
    if args.round2_iterations > 0:
        stacked = torch.stack(pool_patterns, dim=0) if pool_patterns else None
        if stacked is not None:
            active_frac = (stacked > 0).float().mean(dim=0)
            empirically_variable = ((active_frac > 0.001) & (active_frac < 0.999)).nonzero(as_tuple=True)[0]
            round2_candidates = (
                all_indices[torch.isin(all_indices, empirically_variable)]
                if empirically_variable.numel() else all_indices
            )
            if round2_candidates.numel() == 0:
                print("[PatternSearchPGD] Round 1 found no neurons that ever varied; round 2 falls back to "
                      "round 1's full candidate set.")
                round2_candidates = all_indices
            print(
                f"[PatternSearchPGD] Round 1 empirically found {round2_candidates.numel()}/{all_indices.numel()} "
                f"candidate neurons that actually flip at least once -- round 2 restricts flip candidates to "
                f"just these for {args.round2_iterations} more attempts."
            )
            round2_attempts = collect_round("round2", args.round2_iterations, args.collect_timeout, round2_candidates)
            round2_pool_size = len(pool_points)
            print(
                f"[PatternSearchPGD] Round 2 done: {round2_pool_size - round1_pool_size} new points kept out of "
                f"{round2_attempts} attempts (pool now {round2_pool_size} total)."
            )
            round2_ce_records = attack_phase(pool_points[round1_pool_size:round2_pool_size], "round2")

            # ---------------- Round 3: anchor pull-back (opt-in, needs a round-2 CE) ----------------
            if args.round3_iterations > 0:
                if round2_ce_records:
                    print(
                        f"[PatternSearchPGD] Round 3: gradient-walking near {len(round2_ce_records)} round2 "
                        f"counterexamples (stop once Hamming distance to that anchor's own pattern <= "
                        f"{args.round3_hamming_radius}), {args.round3_iterations} attempts..."
                    )
                    round3_pool_start = len(pool_points)
                    width = (ub - lb).clamp(min=0)
                    step_size = args.pattern_step_size
                    if step_size is None:
                        step_size = max(float(width.abs().max().item()) / max(args.pattern_steps, 1), 1e-6)

                    round_start = time.time()
                    made = 0
                    while made < args.round3_iterations:
                        if time.time() - round_start >= args.collect_timeout:
                            print(f"[PatternSearchPGD] round3 timeout reached after {made} attempts.")
                            break
                        made += 1

                        anchor_idx = random.randrange(len(round2_ce_records))
                        anchor_input, anchor_pattern = round2_ce_records[anchor_idx]

                        # Small random perturbation first: starting exactly at the anchor gives zero
                        # margin-loss gradient (it already matches its own pattern perfectly), so nudge
                        # it off that point before gradient-walking back toward (near) the same pattern.
                        noise = (torch.rand_like(anchor_input) * 2 - 1) * args.round3_noise_scale * width
                        x = torch.max(torch.min(anchor_input + noise, ub), lb)

                        noise_hamming = None
                        steps_used = args.pattern_steps
                        broke_early = False
                        for step_i in range(args.pattern_steps):
                            x_req = x.detach().clone().requires_grad_(True)
                            z_all = _relu_preactivations(wrapped_model, x_req)
                            achieved_sign = torch.where(z_all.detach() > 0, torch.ones_like(z_all), -torch.ones_like(z_all))
                            hamming_now = int((achieved_sign[0] != anchor_pattern).sum().item())
                            if step_i == 0:
                                noise_hamming = hamming_now
                            if hamming_now <= args.round3_hamming_radius:
                                steps_used = step_i
                                broke_early = True
                                break
                            violation = (args.pattern_margin - anchor_pattern.unsqueeze(0) * z_all).clamp(min=0)
                            loss = violation.sum()
                            grad = torch.autograd.grad(loss, x_req)[0]
                            x = x_req.detach() - step_size * torch.sign(grad)
                            x = torch.max(torch.min(x, ub), lb)
                        x = x.detach()

                        achieved_pattern = _relu_sign_pattern(wrapped_model, x)
                        final_hamming_to_anchor = int((achieved_pattern != anchor_pattern).sum().item())
                        # Round 3 deliberately does NOT dedup against the tree: the whole point is to
                        # densely sample the neighborhood of known-good patterns, so near-duplicates are
                        # expected and wanted, not noise to filter out.
                        pool_points.append(x)
                        pool_patterns.append(achieved_pattern)

                        collect_rows.append({
                            "round": "round3", "attempt": made, "kept": True, "pool_size_after": len(pool_points),
                            "noise_hamming": noise_hamming, "steps_used": steps_used, "broke_early": broke_early,
                            "final_hamming_to_anchor": final_hamming_to_anchor,
                        })
                        if args.report_interval > 0 and made % args.report_interval == 0:
                            print(f"[PatternSearchPGD] round3 attempts={made}/{args.round3_iterations}, "
                                  f"pool_size={len(pool_points)}")

                    round3_attempts = made
                    round3_pool_size = len(pool_points)
                    print(
                        f"[PatternSearchPGD] Round 3 done: {round3_pool_size - round3_pool_start} new points kept "
                        f"out of {round3_attempts} attempts (pool now {round3_pool_size} total)."
                    )

                    round3_rows = [r for r in collect_rows if r["round"] == "round3"]
                    if round3_rows:
                        hit = [r["final_hamming_to_anchor"] <= args.round3_hamming_radius for r in round3_rows]
                        hit_steps = sorted(r["steps_used"] for r, h in zip(round3_rows, hit) if h)
                        miss_steps = sorted(r["steps_used"] for r, h in zip(round3_rows, hit) if not h)
                        hit_count = sum(hit)
                        print(
                            f"[PatternSearchPGD] Round 3 hit-rate: {hit_count}/{len(round3_rows)} "
                            f"({100.0 * hit_count / len(round3_rows):.1f}%) reached hamming<="
                            f"{args.round3_hamming_radius} within --pattern-steps={args.pattern_steps}. "
                            f"steps_used -- hits: median={statistics.median(hit_steps) if hit_steps else float('nan'):.0f}, "
                            f"misses: median={statistics.median(miss_steps) if miss_steps else float('nan'):.0f}."
                        )
                    attack_phase(pool_points[round3_pool_start:round3_pool_size], "round3")
                else:
                    print("[PatternSearchPGD] Round 2 found no counterexamples; skipping round 3 (nothing to "
                          "build nearby patterns from).")
        else:
            print("[PatternSearchPGD] Round 1 produced an empty pool; skipping round 2.")

    phase1_time = time.time() - phase1_start
    total_attempts = round1_attempts + round2_attempts + round3_attempts
    counterexamples = all_counterexamples
    phase2_time = sum(stats["time_seconds"] for stats in round_attack_stats.values())
    print(
        f"[PatternSearchPGD] All rounds done: {len(pool_points)} points kept out of {total_attempts} attempts, "
        f"{len(counterexamples)} counterexample(s) total, in {phase1_time:.1f}s."
    )

    collect_csv_path = output_dir / "pattern_search_phase1_collect.csv"
    with open(collect_csv_path, "w", newline="", encoding="utf-8") as f:
        if collect_rows:
            fieldnames: list[str] = []
            for row in collect_rows:
                for key in row.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(collect_rows)
        else:
            f.write("round,attempt,kept,pool_size_after,flip_retry_count\n")

    attack_csv_path = output_dir / "pattern_search_phase2_attack.csv"
    with open(attack_csv_path, "w", newline="", encoding="utf-8") as f:
        if all_attack_rows:
            writer = csv.DictWriter(f, fieldnames=list(all_attack_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_attack_rows)
        else:
            f.write("round,pool_index,is_counterexample,violation_score\n")

    summary = {
        "category": args.category,
        "instance_id": str(instance_id),
        "model_id": str(model_id),
        "total_relu_neurons": total_relu_neurons,
        "round1_attempts": round1_attempts,
        "round1_pool_size": round1_pool_size,
        "round2_attempts": round2_attempts,
        "round2_pool_size": round2_pool_size,
        "round3_attempts": round3_attempts,
        "phase1_attempts": total_attempts,
        "phase1_time_seconds": phase1_time,
        "pool_size": len(pool_points),
        "phase2_time_seconds": phase2_time,
        "counterexamples_found": len(counterexamples),
        "round_attack_stats": round_attack_stats,
        "phase1_csv": str(collect_csv_path),
        "phase2_csv": str(attack_csv_path),
        "params": {k: v for k, v in vars(args).items()},
    }
    summary_path = output_dir / "pattern_search_pgd_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[PatternSearchPGD] Summary written to {summary_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m act.pipeline.fuzzing.pattern_search_pgd",
        description="Two-phase pattern-space PGD: phase 1 collects a diverse pool of points via "
        "margin-loss projections toward randomly-flipped ReLU sign-pattern targets, keeping only ones "
        "whose achieved pattern differs enough from everything already pooled; phase 2 runs real "
        "violation-maximizing PGD on every pooled point and checks for counterexamples.",
    )
    parser.add_argument("--category", default="mnist_fc", help="VNNLIB benchmark category (must be downloaded already).")
    parser.add_argument("--max-instances", type=int, default=1, help="How many instances to load from the category.")
    parser.add_argument("--instance-index", type=int, default=None, help="Restrict to a single loaded instance by index.")
    parser.add_argument(
        "--model-index", type=int, default=0,
        help="Which synthesized wrapped model to use (>0 only occurs for OR-disjunct VNNLIB properties).",
    )
    parser.add_argument(
        "--collect-iterations", type=int, default=200,
        help="Phase 1: number of independent random-flip projection attempts to make.",
    )
    parser.add_argument("--collect-timeout", type=float, default=300.0, help="Wall-clock budget per round, in seconds.")
    parser.add_argument(
        "--pool-target", type=int, default=0,
        help="Stop a collection round early once this many diverse points have been kept overall (0 = disabled).",
    )
    parser.add_argument(
        "--round2-iterations", type=int, default=0,
        help="Round 2: number of additional attempts, using flip candidates restricted to just the neurons "
        "round 1 empirically found to actually flip at least once. 0 (default) = single-round, skip round 2.",
    )
    parser.add_argument(
        "--round3-iterations", type=int, default=0,
        help="Round 3: number of additional attempts (only runs if round 2 found at least one real "
        "counterexample). Each attempt anchors on a random round-2 counterexample's own (input, achieved "
        "pattern), nudges the input by --round3-noise-scale, then gradient-walks back toward the anchor's "
        "own pattern until within --round3-hamming-radius. 0 (default) = skip round 3.",
    )
    parser.add_argument(
        "--round3-hamming-radius", type=int, default=3,
        help="Round 3: stop the gradient walk back toward an anchor's pattern as soon as the achieved "
        "Hamming distance to it is <= this value.",
    )
    parser.add_argument(
        "--round3-noise-scale", type=float, default=0.05,
        help="Round 3: initial random perturbation applied to an anchor counterexample's input, as a "
        "fraction of the input box width per dimension, before gradient-walking back toward its pattern.",
    )
    parser.add_argument(
        "--flip-count", type=int, default=10,
        help="Phase 1: number of ReLU neurons to flip away from each fresh base point's own natural "
        "pattern to build that attempt's projection target.",
    )
    parser.add_argument(
        "--flip-retry-attempts", type=int, default=20,
        help="Phase 1: before projecting, re-pick which neurons to flip (up to this many times) if the "
        "resulting TARGET pattern is already within --min-diversity-hamming of something already "
        "pooled -- a cheap pre-filter so the projection isn't knowingly aimed at a pattern that would "
        "just be discarded afterward.",
    )
    parser.add_argument(
        "--min-diversity-hamming", type=int, default=1,
        help="Phase 1: a projected point is only kept if its achieved activation pattern differs from "
        "EVERY pattern already pooled by at least this many neurons.",
    )
    parser.add_argument("--pattern-steps", type=int, default=10, help="Phase 1: margin-loss projection steps.")
    parser.add_argument("--pattern-margin", type=float, default=0.01, help="Phase 1: target sign-margin.")
    parser.add_argument("--pattern-step-size", type=float, default=None)
    parser.add_argument("--pgd-steps", type=int, default=10, help="Phase 2: attack-loss PGD steps per pooled point.")
    parser.add_argument("--pgd-step-size", type=float, default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for the summary and counterexamples.")
    parser.add_argument("--report-interval", type=int, default=50)
    parser.add_argument("--no-save", action="store_true", help="Don't write counterexample .pt files to disk.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Default: a fresh system-random seed.")
    add_device_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed is None:
        args.seed = secrets.randbits(31)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    initialize_from_args(args)
    run_pattern_search_pgd(args)


if __name__ == "__main__":
    main()
