"""
PatternSearchPGD: two-phase pattern-space PGD attack.

Phase 1 (collect): repeatedly sample a fresh random point in the instance's
input box, build a target ReLU sign pattern by flipping a random subset of
that point's own natural pattern, and run a margin-loss projected-gradient
walk toward that target. Every projected point is kept in a diversity pool
only if its actually-achieved sign pattern differs from every pattern
already pooled by at least --min-diversity-hamming neurons, so the pool ends
up covering many different activation regions instead of near-duplicates of
the same one.

Phase 2 (attack): run real violation-maximizing PGD from every pooled point
and check each result against the instance's OutputSpec for an actual
counterexample.

This deliberately does not hill-climb toward any single target: every phase
1 attempt starts from an independent random point and a freshly randomized
target pattern, trading exploitation for breadth of pattern coverage before
spending any attack-PGD budget.

Usage:
    python -m act.pipeline.fuzzing.pattern_search_pgd \\
        --category mnist_fc --collect-iterations 200 --flip-count 10

Copyright (C) 2025 SVF-tools/ACT
License: AGPLv3+
"""

from __future__ import annotations

import argparse
import json
import random
import secrets
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
        raise RuntimeError(
            f"No VNNLIB specs were loaded for category={args.category!r}. "
            f"Download the benchmark first with: python -m act.pipeline --download {args.category}"
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

    # ---------------- Phase 1: diverse pattern collection ----------------
    pool_points: list[torch.Tensor] = []
    pool_patterns: list[torch.Tensor] = []
    tree = _PatternBKTree()
    # threshold for has_within is "closer than min_diversity_hamming", i.e.
    # reject if some pooled point is within (min_diversity_hamming - 1).
    reject_threshold = max(0, args.min_diversity_hamming - 1)

    phase1_start = time.time()
    attempts = 0
    while attempts < args.collect_iterations:
        if time.time() - phase1_start >= args.collect_timeout:
            print(f"[PatternSearchPGD] Phase 1 timeout reached after {attempts} attempts.")
            break
        if args.pool_target > 0 and len(pool_points) >= args.pool_target:
            print(f"[PatternSearchPGD] Phase 1: pool reached target size {args.pool_target}, stopping early.")
            break
        attempts += 1

        base_point = sample_point()
        natural_pattern = _relu_sign_pattern(wrapped_model, base_point)
        target_pattern = _flip_pattern(natural_pattern, args.flip_count, all_indices)
        for _ in range(args.flip_retry_attempts):
            if not tree.has_within(target_pattern, reject_threshold):
                break
            target_pattern = _flip_pattern(natural_pattern, args.flip_count, all_indices)

        projected = _project_to_pattern(
            base_point, wrapped_model, lb, ub, target_pattern,
            args.pattern_steps, args.pattern_margin, args.pattern_step_size,
        )
        achieved_pattern = _relu_sign_pattern(wrapped_model, projected)
        if not tree.has_within(achieved_pattern, reject_threshold):
            pool_points.append(projected)
            pool_patterns.append(achieved_pattern)
            tree.insert(achieved_pattern)

        if args.report_interval > 0 and attempts % args.report_interval == 0:
            print(
                f"[PatternSearchPGD] phase1 attempts={attempts}/{args.collect_iterations}, "
                f"pool_size={len(pool_points)}"
            )

    phase1_time = time.time() - phase1_start
    print(
        f"[PatternSearchPGD] Phase 1 done: {len(pool_points)} diverse points kept out of "
        f"{attempts} attempts in {phase1_time:.1f}s."
    )

    # ---------------- Phase 2: real attack-loss PGD ----------------
    counterexamples: list[Counterexample] = []
    saved_ce_count = 0
    phase2_start = time.time()

    for i, point in enumerate(pool_points):
        adv_input = _attack_pgd(
            point, wrapped_model, lb, ub, output_spec, label, args.pgd_steps, args.pgd_step_size,
        )
        with torch.no_grad():
            out = wrapped_model(adv_input)
            outputs = out["output"] if isinstance(out, dict) else out
        seeds = FuzzingSeed(
            tensor=adv_input,
            original_tensor=point,
            original_index=torch.zeros(1, dtype=torch.long, device=device),
            label=label,
        )
        violation_mask, batch_ces = checker.check(inputs=adv_input, outputs=outputs, seeds=seeds)
        if bool(violation_mask[0].item()):
            for ce in batch_ces:
                counterexamples.append(ce)
                if not args.no_save:
                    ce.save(output_dir / f"ce_{saved_ce_count + 1}.pt")
                saved_ce_count += 1

        if args.report_interval > 0 and (i + 1) % args.report_interval == 0:
            print(
                f"[PatternSearchPGD] phase2 attacked={i + 1}/{len(pool_points)}, "
                f"counterexamples={len(counterexamples)}"
            )

    phase2_time = time.time() - phase2_start
    print(
        f"[PatternSearchPGD] Phase 2 done: {len(counterexamples)}/{len(pool_points)} counterexamples "
        f"in {phase2_time:.1f}s."
    )

    summary = {
        "category": args.category,
        "instance_id": str(instance_id),
        "model_id": str(model_id),
        "total_relu_neurons": total_relu_neurons,
        "phase1_attempts": attempts,
        "phase1_time_seconds": phase1_time,
        "pool_size": len(pool_points),
        "phase2_time_seconds": phase2_time,
        "counterexamples_found": len(counterexamples),
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
    parser.add_argument("--collect-timeout", type=float, default=300.0, help="Phase 1: wall-clock budget in seconds.")
    parser.add_argument(
        "--pool-target", type=int, default=0,
        help="Phase 1: stop early once this many diverse points have been kept (0 = disabled).",
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
