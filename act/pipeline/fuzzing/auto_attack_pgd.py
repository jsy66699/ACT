"""
AutoAttackPGD: a from-scratch, box-respecting reimplementation of the
four-stage AutoAttack ensemble (Croce & Hein, 2020), used as a strong
ablation baseline for PatternSearchPGD / RandomStartPGD.

Why reimplemented instead of wrapping the official `autoattack` package: the
official APGD/FAB/Square implementations hard-code the valid pixel domain as
`clamp(0., 1.)` with a single scalar `eps`. This project's VNNLIB instances
(e.g. cifar100_2024) define their attack box in *normalized* input space --
lb/ub come straight from the parsed VNNLIB bounds, not `x +/- eps` clipped to
`[0, 1]`. Feeding that into the official package's hard-coded [0, 1] clamp
would silently attack the wrong feasible region. Every attack below instead
projects into the instance's own `(lb, ub)` box, the same box
PatternSearchPGD/RandomStartPGD already use and that has been validated
against the VNNLIB spec.

The four stages, run in order (each stage only runs if every earlier stage
failed to find a genuine counterexample, mirroring AutoAttack's
"stop once broken" design):

1. APGD-CE: untargeted Auto-PGD (momentum + adaptive step-size halving)
   ascending cross-entropy away from the true label.
2. APGD-T: targeted Auto-PGD ascending a DLR-style margin loss toward each of
   the top --n-target-classes classes (ranked by the clean point's own
   logits).
3. FAB-T (simplified): for each target class, iteratively linearizes the
   (z_target - z_true) decision function at the current point, takes the
   closed-form minimal-L1-direction step that would zero it out, then
   extrapolates slightly past the linearized boundary (a stand-in for FAB's
   backward/bias step) and re-checks the *real* (non-linearized) network.
   This is NOT a line-for-line port of the published FAB algorithm (which
   solves an exact L-inf projection onto the linearized boundary via a
   dedicated LP-like routine) -- it captures the same "walk to the nearest
   decision boundary via local linearization" idea at a fraction of the
   implementation/debugging cost.
4. Square Attack (simplified): black-box random search with shrinking square
   patches, greedy acceptance on the same violation score PatternSearchPGD
   uses, seeded with a structured vertical-stripe init as in the original
   paper.

Usage:
    python -m act.pipeline.fuzzing.auto_attack_pgd \\
        --category cifar100_2024 --apgd-steps 50 --n-target-classes 5
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import secrets
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from act.pipeline.fuzzing.checker import PropertyChecker
from act.pipeline.fuzzing.corpus import FuzzingSeed
from act.pipeline.fuzzing.pattern_search_pgd import (
    _auto_pgd_core,
    _dlr_loss_targeted,
    _forward,
    _project_box,
    _violation_score,
    load_instance_for_attack,
)
from act.util.cli_utils import add_device_args, add_sam2_mask_args, initialize_from_args
from act.util.path_config import get_pipeline_log_dir, get_project_root

DEFAULT_OUTPUT_DIR = Path(get_pipeline_log_dir()) / "auto_attack_pgd"
DEFAULT_CONFIG_PATH = Path(get_project_root()) / "act" / "config" / "auto_attack_pgd.yaml"

_YAML_CONFIG_DEFAULTS = {
    "category": "mnist_fc",
    "max_instances": 1,
    "instance_index": None,
    "model_index": 0,
    "restarts": 1,
    "apgd_steps": 50,
    "apgd_restarts": 1,
    "n_target_classes": 5,
    "fab_steps": 30,
    "fab_bias": 0.10,
    "square_queries": 500,
    "report_interval": 1,
}


def _load_yaml_config(config_path: Optional[str]) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_yaml_args(args: argparse.Namespace) -> None:
    """CLI flag > --config YAML > code default, same rule as PatternSearchPGD/RandomStartPGD."""
    yaml_data = _load_yaml_config(args.config)
    for key, fallback in _YAML_CONFIG_DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, yaml_data.get(key, fallback))


# ---------------- Stage 3: FAB-style simplified boundary attack ----------------

def _fab_style(
    x0: torch.Tensor, model: nn.Module, lb: torch.Tensor, ub: torch.Tensor,
    label: torch.Tensor, target_classes: list[int], steps: int, bias: float,
    check_fn: Callable[[torch.Tensor], bool],
) -> Optional[torch.Tensor]:
    """For each target class, linearize g(x) = z_true(x) - z_target(x) at the
    current point and take the closed-form minimal-L1-norm step (i.e. the
    L-inf-dual step) that would zero the linear approximation, then push a
    further `bias` fraction past it before re-clamping to the box -- a stand-in
    for FAB's exact backward/extrapolation step. Checks the real network after
    every step (not just the linear approximation) and returns immediately on
    a genuine violation."""
    for t in target_classes:
        x = x0.detach().clone()
        for _ in range(steps):
            x_req = x.detach().clone().requires_grad_(True)
            z = _forward(model, x_req)[0]
            g = z[label[0]] - z[t]
            grad = torch.autograd.grad(g, x_req)[0]
            denom = grad.abs().sum().clamp(min=1e-12)
            step = -(g.detach() / denom) * torch.sign(grad) * (1.0 + bias)
            x = _project_box(x.detach() + step, lb, ub)
            if check_fn(x):
                return x.detach()
    return None


# ---------------- Stage 4: Square Attack (simplified) ----------------

def _square_attack(
    x0: torch.Tensor, model: nn.Module, lb: torch.Tensor, ub: torch.Tensor,
    label: torch.Tensor, output_spec: Any, n_queries: int,
    check_fn: Callable[[torch.Tensor], bool], report_interval: int,
) -> Optional[torch.Tensor]:
    """Black-box random search (Andriushchenko et al., 2020): greedily accept
    randomly-placed square patches (side length shrinking over the query
    budget per the original paper's schedule) whenever they improve the
    violation score, seeded with a structured vertical-stripe init."""
    half_width = (ub - lb) / 2
    H, W = x0.shape[-2], x0.shape[-1]

    x_adv = x0.detach().clone()
    stripe = max(1, W // 8)
    for c0 in range(0, W, stripe):
        sign = 1.0 if random.random() < 0.5 else -1.0
        x_adv[..., :, c0:c0 + stripe] = x0[..., :, c0:c0 + stripe] + sign * half_width[..., :, c0:c0 + stripe]
    x_adv = _project_box(x_adv, lb, ub)
    if check_fn(x_adv):
        return x_adv

    with torch.no_grad():
        best_loss = float(_violation_score(output_spec, _forward(model, x_adv), label)[0].item())

    schedule = ((0.00, 0.30), (0.05, 0.20), (0.20, 0.10), (0.40, 0.07), (0.60, 0.05), (0.80, 0.03))
    for q in range(n_queries):
        frac = q / max(n_queries, 1)
        p = schedule[0][1]
        for thresh, val in schedule:
            if frac >= thresh:
                p = val
        side = max(1, min(H, W, int(round((p ** 0.5) * min(H, W)))))
        r0 = random.randint(0, max(0, H - side))
        c0 = random.randint(0, max(0, W - side))

        candidate = x_adv.detach().clone()
        sign = 1.0 if random.random() < 0.5 else -1.0
        region = (Ellipsis, slice(r0, r0 + side), slice(c0, c0 + side))
        candidate[region] = _project_box(
            x0[region] + sign * half_width[region], lb[region], ub[region],
        )

        with torch.no_grad():
            loss = float(_violation_score(output_spec, _forward(model, candidate), label)[0].item())
        if loss > best_loss:
            best_loss = loss
            x_adv = candidate
            if check_fn(x_adv):
                return x_adv

        if report_interval > 0 and (q + 1) % report_interval == 0:
            print(f"[AutoAttackPGD] square query {q + 1}/{n_queries}, best_violation_score={best_loss:.4f}")

    return None


def run_one_attempt(loaded, checker: PropertyChecker, args: argparse.Namespace) -> dict[str, Any]:
    wrapped_model = loaded.wrapped_model
    output_spec = loaded.output_spec
    lb, ub = loaded.lb, loaded.ub
    label = loaded.label
    device = loaded.device
    x0 = (lb + ub) / 2  # approximates the clean/original image the VNNLIB box was built around

    found_ce: list[Any] = []

    def check_fn(x: torch.Tensor) -> bool:
        with torch.no_grad():
            out = _forward(wrapped_model, x)
        seeds = FuzzingSeed(
            tensor=x, original_tensor=x0, original_index=torch.zeros(1, dtype=torch.long, device=device), label=label,
        )
        violation_mask, ces = checker.check(inputs=x, outputs=out, seeds=seeds)
        if bool(violation_mask[0].item()):
            found_ce.extend(ces)
            return True
        return False

    def random_init() -> torch.Tensor:
        return _project_box(x0 + (torch.rand_like(x0) * 2 - 1) * (ub - lb) * 0.5, lb, ub)

    # ---------------- Stage 1: APGD-CE ----------------
    print("[AutoAttackPGD] Stage 1/4: APGD-CE (untargeted)...")
    for r in range(max(1, args.apgd_restarts)):
        x_init = x0.detach().clone() if r == 0 else random_init()
        x_cand = _auto_pgd_core(x_init, wrapped_model, lb, ub, lambda out: F.cross_entropy(out, label), args.apgd_steps)
        if check_fn(x_cand):
            return {"stage": "apgd-ce", "counterexamples": found_ce}
    print("[AutoAttackPGD] Stage 1/4: no counterexample.")

    with torch.no_grad():
        logits0 = _forward(wrapped_model, x0)[0].clone()
    logits0[label[0]] = float("-inf")
    k = min(args.n_target_classes, logits0.numel() - 1)
    target_classes = torch.topk(logits0, k=k).indices.tolist()

    # ---------------- Stage 2: APGD-T ----------------
    print(f"[AutoAttackPGD] Stage 2/4: APGD-T (targeted, classes={target_classes})...")
    for t in target_classes:
        for r in range(max(1, args.apgd_restarts)):
            x_init = x0.detach().clone() if r == 0 else random_init()
            x_cand = _auto_pgd_core(
                x_init, wrapped_model, lb, ub, lambda out, t=t: _dlr_loss_targeted(out, label, t), args.apgd_steps,
            )
            if check_fn(x_cand):
                return {"stage": "apgd-t", "counterexamples": found_ce}
    print("[AutoAttackPGD] Stage 2/4: no counterexample.")

    # ---------------- Stage 3: FAB-T (simplified) ----------------
    print("[AutoAttackPGD] Stage 3/4: FAB-T (simplified boundary search)...")
    if _fab_style(x0, wrapped_model, lb, ub, label, target_classes, args.fab_steps, args.fab_bias, check_fn) is not None:
        return {"stage": "fab-t", "counterexamples": found_ce}
    print("[AutoAttackPGD] Stage 3/4: no counterexample.")

    # ---------------- Stage 4: Square Attack (simplified) ----------------
    print(f"[AutoAttackPGD] Stage 4/4: Square Attack ({args.square_queries} queries)...")
    if _square_attack(
        x0, wrapped_model, lb, ub, label, output_spec, args.square_queries, check_fn, args.report_interval,
    ) is not None:
        return {"stage": "square", "counterexamples": found_ce}
    print("[AutoAttackPGD] Stage 4/4: no counterexample.")

    return {"stage": None, "counterexamples": []}


def run_auto_attack_pgd(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_instance_for_attack(args, "AutoAttackPGD")
    checker = PropertyChecker(loaded.output_spec)
    print(f"[AutoAttackPGD] model={loaded.model_id}, restarts={args.restarts}")

    rows: list[dict[str, Any]] = []
    ce_count = 0
    saved_ce_count = 0
    stage_counts: dict[str, int] = {"apgd-ce": 0, "apgd-t": 0, "fab-t": 0, "square": 0}

    start_time = time.time()
    for i in range(args.restarts):
        print(f"[AutoAttackPGD] ===== restart {i + 1}/{args.restarts} =====")
        result = run_one_attempt(loaded, checker, args)
        stage = result["stage"]
        is_ce = stage is not None
        rows.append({"restart": i, "is_counterexample": is_ce, "stage_found": stage})
        if is_ce:
            ce_count += 1
            stage_counts[stage] += 1
            for ce in result["counterexamples"]:
                if not args.no_save:
                    ce.save(output_dir / f"ce_{saved_ce_count + 1}.pt")
                saved_ce_count += 1
        print(f"[AutoAttackPGD] restart {i + 1}/{args.restarts} result: "
              f"{'BROKEN at ' + stage if is_ce else 'not broken'}")

    elapsed = time.time() - start_time
    rate = 100.0 * ce_count / args.restarts if args.restarts else 0.0
    print(f"[AutoAttackPGD] Done: {ce_count}/{args.restarts} counterexamples ({rate:.1f}%) in {elapsed:.1f}s. "
          f"Stage breakdown: {stage_counts}")

    attack_csv_path = output_dir / "auto_attack_pgd_attack.csv"
    with open(attack_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["restart", "is_counterexample", "stage_found"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "method": "auto_attack_pgd",
        "category": args.category,
        "instance_id": str(loaded.instance_id),
        "model_id": str(loaded.model_id),
        "restarts": args.restarts,
        "counterexamples_found": ce_count,
        "counterexample_rate": rate,
        "stage_counts": stage_counts,
        "time_seconds": elapsed,
        "attack_csv": str(attack_csv_path),
        "params": {k: v for k, v in vars(args).items()},
    }
    summary_path = output_dir / "auto_attack_pgd_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[AutoAttackPGD] Summary written to {summary_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m act.pipeline.fuzzing.auto_attack_pgd",
        description="Ablation baseline for PatternSearchPGD: a from-scratch, box-respecting reimplementation "
        "of the 4-stage AutoAttack ensemble (APGD-CE, APGD-T, simplified FAB-T, simplified Square Attack), "
        "run in order with early stopping once any stage finds a genuine counterexample.",
    )
    parser.add_argument(
        "--config", default=None,
        help=f"Path to a run-selection YAML (category/max_instances/instance_index/model_index/restarts/"
        f"apgd_steps/apgd_restarts/n_target_classes/fab_steps/fab_bias/square_queries/report_interval). "
        f"Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument("--category", default=None, help="VNNLIB benchmark category. Default: from --config YAML.")
    parser.add_argument("--max-instances", type=int, default=None, help="Default: from --config YAML.")
    parser.add_argument("--instance-index", type=int, default=None, help="Default: from --config YAML.")
    parser.add_argument("--model-index", type=int, default=None, help="Default: from --config YAML.")
    parser.add_argument(
        "--restarts", type=int, default=None,
        help="Number of independent full 4-stage pipeline attempts. Default: from --config YAML.",
    )
    parser.add_argument(
        "--apgd-steps", type=int, default=None,
        help="Auto-PGD iterations per APGD-CE/APGD-T run. Default: from --config YAML.",
    )
    parser.add_argument(
        "--apgd-restarts", type=int, default=None,
        help="Random-init restarts per APGD-CE call and per APGD-T target class. Default: from --config YAML.",
    )
    parser.add_argument(
        "--n-target-classes", type=int, default=None,
        help="Number of top (by clean-point logit) classes targeted by APGD-T/FAB-T. Default: from --config YAML.",
    )
    parser.add_argument("--fab-steps", type=int, default=None, help="FAB-T iterations per target class. Default: from --config YAML.")
    parser.add_argument(
        "--fab-bias", type=float, default=None,
        help="Fraction to overshoot past FAB-T's linearized boundary each step. Default: from --config YAML.",
    )
    parser.add_argument("--square-queries", type=int, default=None, help="Square Attack query budget. Default: from --config YAML.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--report-interval", type=int, default=None, help="Default: from --config YAML.")
    parser.add_argument("--no-save", action="store_true", help="Don't write counterexample .pt files to disk.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Default: a fresh system-random seed.")
    add_device_args(parser)
    add_sam2_mask_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _resolve_yaml_args(args)
    if args.seed is None:
        args.seed = secrets.randbits(31)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    initialize_from_args(args)
    run_auto_attack_pgd(args)


if __name__ == "__main__":
    main()
