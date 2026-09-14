"""
RandomStartPGD: plain random-restart PGD, as an ablation baseline for
PatternSearchPGD.

Every restart is independent: sample a fresh uniform-random point inside the
instance's input box (no ReLU-pattern targeting, no diversity pool, no
phase-1/phase-2 split -- just a random start), then run the same
violation-maximizing PGD attack PatternSearchPGD uses in its phase 2, and
check the result for an actual counterexample.

This isolates how much of PatternSearchPGD's counterexample yield comes from
its pattern-space diversity search versus from attack-loss PGD alone: same
attack loss, same PGD steps, same instance/model loading and label
resolution -- the only difference is the starting point distribution.

Usage:
    python -m act.pipeline.fuzzing.random_start_pgd \\
        --category cifar100_2024 --restarts 500 --pgd-steps 50
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import secrets
import time
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

from act.pipeline.fuzzing.checker import PropertyChecker
from act.pipeline.fuzzing.corpus import FuzzingSeed
from act.pipeline.fuzzing.pattern_search_pgd import (
    _attack_apgd_ce,
    _attack_apgd_ce_bin,
    _attack_apgd_ce_t,
    _attack_apgd_cw,
    _attack_apgd_dlr,
    _attack_apgd_t,
    _attack_pgd,
    _attack_pgd_ce_bin,
    _hamming,
    _PatternBKTree,
    _relu_sign_pattern,
    _violation_score,
    load_instance_for_attack,
)
from act.util.cli_utils import add_device_args, add_sam2_mask_args, initialize_from_args
from act.util.path_config import get_pipeline_log_dir, get_project_root

DEFAULT_OUTPUT_DIR = Path(get_pipeline_log_dir()) / "random_start_pgd"
DEFAULT_CONFIG_PATH = Path(get_project_root()) / "act" / "config" / "random_start_pgd.yaml"

_YAML_CONFIG_DEFAULTS = {
    "category": "mnist_fc",
    "max_instances": 1,
    "instance_index": None,
    "model_index": 0,
    "restarts": 200,
    "report_interval": 50,
    "pgd_steps": 10,
    "pgd_step_size": None,
    "attack_strategy": "pgd",
    "apgd_t_target_classes": 5,
    "ce_bin_target_class": 72,
    "min_diversity_hamming": 1,
}


def _load_yaml_config(config_path: Optional[str]) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_yaml_args(args: argparse.Namespace) -> None:
    """Fill in any of _YAML_CONFIG_DEFAULTS' keys left unset (None) on the
    command line, in order: CLI flag > --config YAML > code default."""
    yaml_data = _load_yaml_config(args.config)
    for key, fallback in _YAML_CONFIG_DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, yaml_data.get(key, fallback))


def run_random_start_pgd(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_instance_for_attack(args, "RandomStartPGD")
    wrapped_model = loaded.wrapped_model
    output_spec = loaded.output_spec
    lb, ub = loaded.lb, loaded.ub
    label = loaded.label
    device = loaded.device

    checker = PropertyChecker(output_spec)

    def sample_point() -> torch.Tensor:
        return (lb + torch.rand_like(lb) * (ub - lb).clamp(min=0)).detach()

    print(f"[RandomStartPGD] model={loaded.model_id}, restarts={args.restarts}, pgd_steps={args.pgd_steps}, "
          f"attack_strategy={args.attack_strategy}")

    # ---------------- seed pool diversity tracking ----------------
    # RandomStartPGD has no deliberate diversity search (unlike PatternSearchPGD's
    # phase-1 pool), but every restart's random starting point still lands on some
    # concrete ReLU sign pattern. Tracking those with the exact same BK-tree
    # dedup rule PatternSearchPGD's pool uses (min_diversity_hamming) gives a
    # directly comparable "seed pool" stat: how much activation-pattern diversity
    # plain uniform random sampling covers on its own, for the same attempt
    # budget, versus PatternSearchPGD's deliberately-projected pool.
    pool_tree = _PatternBKTree()
    pool_patterns: list[torch.Tensor] = []
    reject_threshold = max(0, args.min_diversity_hamming - 1)

    attack_rows: list[dict[str, Any]] = []
    counterexamples = []
    saved_ce_count = 0
    ce_count = 0

    start_time = time.time()
    for i in range(args.restarts):
        x0 = sample_point()
        seed_pattern = _relu_sign_pattern(wrapped_model, x0)
        seed_pool_kept = not pool_tree.has_within(seed_pattern, reject_threshold)
        if seed_pool_kept:
            pool_patterns.append(seed_pattern)
            pool_tree.insert(seed_pattern)
        if args.attack_strategy == "apgd_t":
            adv_input = _attack_apgd_t(
                x0, wrapped_model, lb, ub, output_spec, label, args.pgd_steps,
                args.apgd_t_target_classes, checker, device,
            )
        elif args.attack_strategy == "apgd_ce":
            adv_input = _attack_apgd_ce(x0, wrapped_model, lb, ub, label, args.pgd_steps)
        elif args.attack_strategy == "apgd_ce_t":
            adv_input = _attack_apgd_ce_t(
                x0, wrapped_model, lb, ub, output_spec, label, args.pgd_steps,
                args.apgd_t_target_classes, checker, device,
            )
        elif args.attack_strategy == "apgd_dlr":
            adv_input = _attack_apgd_dlr(x0, wrapped_model, lb, ub, label, args.pgd_steps)
        elif args.attack_strategy == "apgd_cw":
            adv_input = _attack_apgd_cw(x0, wrapped_model, lb, ub, label, args.pgd_steps)
        elif args.attack_strategy == "apgd_ce_bin":
            adv_input = _attack_apgd_ce_bin(x0, wrapped_model, lb, ub, label, args.ce_bin_target_class, args.pgd_steps)
        elif args.attack_strategy == "pgd_ce_bin":
            adv_input = _attack_pgd_ce_bin(
                x0, wrapped_model, lb, ub, label, args.ce_bin_target_class, args.pgd_steps, args.pgd_step_size,
            )
        else:
            adv_input = _attack_pgd(x0, wrapped_model, lb, ub, output_spec, label, args.pgd_steps, args.pgd_step_size)
        with torch.no_grad():
            out = wrapped_model(adv_input)
            outputs = out["output"] if isinstance(out, dict) else out
        seeds = FuzzingSeed(
            tensor=adv_input, original_tensor=x0,
            original_index=torch.zeros(1, dtype=torch.long, device=device), label=label,
        )
        violation_mask, batch_ces = checker.check(inputs=adv_input, outputs=outputs, seeds=seeds)
        is_ce = bool(violation_mask[0].item())
        score = float(_violation_score(output_spec, outputs, label)[0].item())

        attack_rows.append({
            "restart": i, "is_counterexample": is_ce, "violation_score": score,
            "seed_pool_kept": seed_pool_kept,
        })
        if is_ce:
            ce_count += 1
            for ce in batch_ces:
                counterexamples.append(ce)
                if not args.no_save:
                    ce.save(output_dir / f"ce_{saved_ce_count + 1}.pt")
                saved_ce_count += 1

        if args.report_interval > 0 and (i + 1) % args.report_interval == 0:
            print(f"[RandomStartPGD] restart {i + 1}/{args.restarts}, counterexamples={ce_count}, "
                  f"seed_pool_size={len(pool_patterns)}")

    elapsed = time.time() - start_time
    rate = 100.0 * ce_count / args.restarts if args.restarts else 0.0
    print(f"[RandomStartPGD] Done: {ce_count}/{args.restarts} counterexamples ({rate:.1f}%) in {elapsed:.1f}s.")

    # Pairwise diversity of the seed pool, for direct comparison against
    # PatternSearchPGD's phase-1 pool (same sampling-based estimate it uses).
    seed_pool_size = len(pool_patterns)
    seed_pool_pairwise_hamming_mean = None
    if seed_pool_size >= 2:
        n_pairs = min(2000, seed_pool_size * (seed_pool_size - 1) // 2)
        dists = []
        for _ in range(n_pairs):
            i_idx, j_idx = random.randrange(seed_pool_size), random.randrange(seed_pool_size)
            if i_idx != j_idx:
                dists.append(_hamming(pool_patterns[i_idx], pool_patterns[j_idx]))
        if dists:
            seed_pool_pairwise_hamming_mean = sum(dists) / len(dists)
    print(f"[RandomStartPGD] Seed pool: {seed_pool_size} distinct pattern(s) out of {args.restarts} restarts "
          f"(min_diversity_hamming={args.min_diversity_hamming}, pairwise_hamming_mean="
          f"{seed_pool_pairwise_hamming_mean if seed_pool_pairwise_hamming_mean is None else f'{seed_pool_pairwise_hamming_mean:.1f}'}).")

    attack_csv_path = output_dir / "random_start_pgd_attack.csv"
    with open(attack_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["restart", "is_counterexample", "violation_score", "seed_pool_kept"])
        writer.writeheader()
        writer.writerows(attack_rows)

    summary = {
        "method": "random_start_pgd",
        "category": args.category,
        "instance_id": str(loaded.instance_id),
        "model_id": str(loaded.model_id),
        "restarts": args.restarts,
        "counterexamples_found": ce_count,
        "counterexample_rate": rate,
        "time_seconds": elapsed,
        "seed_pool_size": seed_pool_size,
        "seed_pool_pairwise_hamming_mean": seed_pool_pairwise_hamming_mean,
        "attack_csv": str(attack_csv_path),
        "params": {k: v for k, v in vars(args).items()},
    }
    summary_path = output_dir / "random_start_pgd_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[RandomStartPGD] Summary written to {summary_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m act.pipeline.fuzzing.random_start_pgd",
        description="Ablation baseline for PatternSearchPGD: plain random-restart PGD, no ReLU-pattern "
        "targeting or diversity pool -- just a fresh uniform-random start in the input box followed by "
        "the same violation-maximizing attack PGD, repeated --restarts times.",
    )
    parser.add_argument(
        "--config", default=None,
        help=f"Path to a run-selection YAML (category/max_instances/instance_index/model_index/restarts/"
        f"report_interval/pgd_steps/pgd_step_size/attack_strategy/apgd_t_target_classes/"
        f"min_diversity_hamming). Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument("--category", default=None, help="VNNLIB benchmark category. Default: from --config YAML.")
    parser.add_argument("--max-instances", type=int, default=None, help="Default: from --config YAML.")
    parser.add_argument("--instance-index", type=int, default=None, help="Default: from --config YAML.")
    parser.add_argument("--model-index", type=int, default=None, help="Default: from --config YAML.")
    parser.add_argument(
        "--restarts", type=int, default=None,
        help="Number of independent random-start PGD attempts. Default: from --config YAML.",
    )
    parser.add_argument("--pgd-steps", type=int, default=None, help="PGD steps per restart. Default: from --config YAML.")
    parser.add_argument(
        "--pgd-step-size", type=float, default=None,
        help="PGD step size (null/omitted = auto per-instance). Only used when --attack-strategy is 'pgd' "
        "or 'pgd_ce_bin' (the two fixed-step, non-Auto-PGD strategies). Default: from --config YAML.",
    )
    parser.add_argument(
        "--attack-strategy",
        choices=["pgd", "apgd_t", "apgd_ce", "apgd_ce_t", "apgd_dlr", "apgd_cw", "apgd_ce_bin", "pgd_ce_bin"],
        default=None,
        help="'pgd' (default) is the plain violation-loss PGD above; 'apgd_ce' replaces it with untargeted "
        "Auto-PGD (cross-entropy loss, momentum + adaptive step size); 'apgd_dlr' is untargeted Auto-PGD "
        "ascending the untargeted DLR margin instead of CE (tracks violation_score more directly, without "
        "committing to one target class); 'apgd_cw' is untargeted Auto-PGD ascending the raw (unnormalized) "
        "CW margin -- the same quantity 'pgd' ascends with fixed-step sign-PGD, isolating whether Auto-PGD's "
        "momentum/step adaptation helps on its own; 'apgd_t' replaces it with targeted Auto-PGD (DLR loss, "
        "same momentum/step-size machinery) against each restart's own top --apgd-t-target-classes classes; "
        "'apgd_ce_t' is the same single-target-class structure as apgd_t but ascending "
        "-cross_entropy(out, t) per candidate target instead of DLR; 'apgd_ce_bin' is a pure two-class "
        "attack -- softmax CE over ONLY {true class, --ce-bin-target-class}, every other class left out of "
        "the loss entirely, with the SAME fixed target class used for every restart (no per-restart top-k "
        "search); 'pgd_ce_bin' is apgd_ce_bin's plain-PGD counterpart -- same fixed-step sign-PGD loop as "
        "'pgd' (uses --pgd-step-size, no Auto-PGD momentum), ascending the same binary-CE loss. Every "
        "other apgd_* mode reuses --pgd-steps as the Auto-PGD iteration budget (--pgd-step-size is ignored "
        "for those). Default: from --config YAML.",
    )
    parser.add_argument(
        "--apgd-t-target-classes", type=int, default=None,
        help="Only used when --attack-strategy=apgd_t or apgd_ce_t: how many of each restart's own top "
        "classes (by its current logits) to try as targets. Default: from --config YAML.",
    )
    parser.add_argument(
        "--ce-bin-target-class", type=int, default=None,
        help="Only used when --attack-strategy is apgd_ce_bin or pgd_ce_bin: the fixed target class every "
        "restart is attacked toward (the other half of the binary CE, alongside the instance's true "
        "label). Default: from --config YAML (72 -- instance 11's observed true=73 -> misclassified-as=72 "
        "direction).",
    )
    parser.add_argument(
        "--min-diversity-hamming", type=int, default=None,
        help="Reporting only (doesn't affect which restarts get attacked): a restart's starting-point ReLU "
        "sign pattern only counts toward the seed-pool stats in the summary/CSV if it differs from every "
        "pattern already counted by at least this many neurons -- the same rule PatternSearchPGD's phase-1 "
        "pool uses, so seed_pool_size is directly comparable to PatternSearchPGD's pool_size for the same "
        "attempt budget. Default: from --config YAML.",
    )
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
    run_random_start_pgd(args)


if __name__ == "__main__":
    main()
