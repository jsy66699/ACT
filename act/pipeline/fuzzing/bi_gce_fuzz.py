"""
BiGceFuzz: run ACTFuzzer against a single VNNLIB instance, selected the same
way PatternSearchPGD/RandomStartPGD/AutoAttackPGD select theirs
(--category/--max-instances/--instance-index/--model-index, CLI > this
script's own --config YAML > code default).

This differs from `python -m act.pipeline --fuzz`, which loads and batches
together up to --max-instances instances into ONE fuzzing run. BiGceFuzz
loads exactly one instance and fuzzes it alone -- useful for isolating a
single hard instance (e.g. one PatternSearchPGD/RandomStartPGD couldn't
break) and pointing ACTFuzzer's BI/GCE state-driven search at it specifically.

BI/GCE itself (enable_bi_gce, mutation_weights, PatternStateManager tuning,
...) is NOT reconfigured here -- it still comes from act/config/pipeline.yaml's
`fuzzing:` section via FuzzingConfig.from_yaml(), same as every other
ACTFuzzer entrypoint. This script's own YAML only owns instance selection and
this run's budget (iterations/timeout/report_interval), so there's exactly
one place to look for "which BI/GCE knobs are on" (pipeline.yaml) and one
place for "which instance am I pointing it at" (this script's own config).

Usage:
    python -m act.pipeline.fuzzing.bi_gce_fuzz \\
        --category cifar100_2024 --instance-index 11 --iterations 5000
"""

from __future__ import annotations

import argparse
import json
import random
import secrets
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

from act.front_end.spec_creator_base import LabeledInputTensor
from act.pipeline.fuzzing.actfuzzer import ACTFuzzer, FuzzingConfig
from act.pipeline.fuzzing.pattern_search_pgd import load_instance_for_attack
from act.util.cli_utils import add_device_args, initialize_from_args
from act.util.path_config import get_pipeline_log_dir, get_project_root

DEFAULT_OUTPUT_DIR = Path(get_pipeline_log_dir()) / "bi_gce_fuzz"
DEFAULT_CONFIG_PATH = Path(get_project_root()) / "act" / "config" / "bi_gce_fuzz.yaml"

_YAML_CONFIG_DEFAULTS = {
    "category": "mnist_fc",
    "max_instances": 1,
    "instance_index": None,
    "model_index": 0,
    "iterations": 10000,
    "timeout": 3600.0,
    "report_interval": 500,
}


def _load_yaml_config(config_path: Optional[str]) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_yaml_args(args: argparse.Namespace) -> None:
    """CLI flag > --config YAML > code default, same rule as
    PatternSearchPGD/RandomStartPGD/AutoAttackPGD."""
    yaml_data = _load_yaml_config(args.config)
    for key, fallback in _YAML_CONFIG_DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, yaml_data.get(key, fallback))


def run_bi_gce_fuzz(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_instance_for_attack(args, "BiGceFuzz")
    if not loaded.labeled_tensors:
        raise RuntimeError("No LabeledInputTensor(s) available for this instance -- ACTFuzzer needs at "
                            "least one to seed its corpus.")

    # labeled_tensors[0].label is None for VNNLIB 2.0 instances (e.g.
    # cifar100_2024) that don't carry an explicit label alongside the input
    # tensor -- load_instance_for_attack already resolved the real one
    # (loaded.label, from output_spec.y_true when needed). Passing the raw
    # labeled_tensors straight to ACTFuzzer would silently give every seed
    # label=-1, which PropertyChecker's valid_mask treats as "no label" and
    # rejects every violation -- same bug already fixed once this session in
    # pattern_search_pgd.py's own label resolution.
    initial_seeds = [
        LabeledInputTensor(tensor=lt.tensor, label=loaded.label.to(dtype=torch.long))
        for lt in loaded.labeled_tensors
    ]

    config = FuzzingConfig.from_yaml(
        max_iterations=args.iterations,
        timeout_seconds=args.timeout,
        save_counterexamples=not args.no_save,
        output_dir=output_dir,
        report_interval=args.report_interval,
    )
    print(f"[BiGceFuzz] model={loaded.model_id}")
    print(f"[BiGceFuzz] enable_bi_gce={config.enable_bi_gce}, admission_mode={config.admission_mode!r}, "
          f"scheduling_mode={config.scheduling_mode!r} (from act/config/pipeline.yaml)")
    print(f"[BiGceFuzz] max_iterations={config.max_iterations}, timeout_seconds={config.timeout_seconds}")

    fuzzer = ACTFuzzer(
        wrapped_model=loaded.wrapped_model,
        initial_seeds=initial_seeds,
        config=config,
    )
    report = fuzzer.fuzz()
    report.save(output_dir)

    source_counts: dict[str, int] = {}
    for ce in report.counterexamples:
        key = ce.source or "unknown"
        source_counts[key] = source_counts.get(key, 0) + 1

    print(f"[BiGceFuzz] Done: {len(report.counterexamples)} counterexample(s) in {report.total_iterations} "
          f"iteration(s), {report.total_time:.1f}s, neuron_coverage={report.neuron_coverage:.2%}.")
    print(f"[BiGceFuzz] By source: {source_counts}")

    summary = {
        "method": "bi_gce_fuzz",
        "category": args.category,
        "instance_id": str(loaded.instance_id),
        "model_id": str(loaded.model_id),
        "enable_bi_gce": config.enable_bi_gce,
        "admission_mode": config.admission_mode,
        "scheduling_mode": config.scheduling_mode,
        "iterations": report.total_iterations,
        "time_seconds": report.total_time,
        "counterexamples_found": len(report.counterexamples),
        "counterexamples_by_source": source_counts,
        "last_hpgd_natural_to_target_hamming": fuzzer.last_hpgd_natural_to_target_hamming,
        "last_hpgd_natural_to_achieved_hamming": fuzzer.last_hpgd_natural_to_achieved_hamming,
        "last_hpgd_target_to_achieved_hamming": fuzzer.last_hpgd_target_to_achieved_hamming,
        "last_hpgd_on_target_hit_rate": fuzzer.last_hpgd_on_target_hit_rate,
        "last_hpgd_on_target_hit_count": fuzzer.last_hpgd_on_target_hit_count,
        "last_hpgd_on_target_total": fuzzer.last_hpgd_on_target_total,
        "neuron_coverage": report.neuron_coverage,
        "mutations": report.total_mutations,
        "seeds_explored": report.seeds_explored,
        "num_of_never_activated_neurons": report.num_of_never_activated_neurons,
        "params": {k: v for k, v in vars(args).items()},
    }
    summary_path = output_dir / "bi_gce_fuzz_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[BiGceFuzz] Summary written to {summary_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m act.pipeline.fuzzing.bi_gce_fuzz",
        description="Run ACTFuzzer (BI/GCE state-driven search, configured via act/config/pipeline.yaml's "
        "fuzzing.enable_bi_gce) against a single VNNLIB instance, selected the same way "
        "PatternSearchPGD/RandomStartPGD/AutoAttackPGD select theirs.",
    )
    parser.add_argument(
        "--config", default=None,
        help=f"Path to a run-selection YAML (category/max_instances/instance_index/model_index/iterations/"
        f"timeout/report_interval). Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument("--category", default=None, help="VNNLIB benchmark category. Default: from --config YAML.")
    parser.add_argument("--max-instances", type=int, default=None, help="Default: from --config YAML.")
    parser.add_argument("--instance-index", type=int, default=None, help="Default: from --config YAML.")
    parser.add_argument("--model-index", type=int, default=None, help="Default: from --config YAML.")
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="Max fuzzing iterations (FuzzingConfig.max_iterations override). Default: from --config YAML.",
    )
    parser.add_argument(
        "--timeout", type=float, default=None,
        help="Wall-clock time budget in seconds (FuzzingConfig.timeout_seconds override). "
        "Default: from --config YAML.",
    )
    parser.add_argument("--report-interval", type=int, default=None, help="Default: from --config YAML.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--no-save", action="store_true", help="Don't write counterexample .pt files to disk.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Default: a fresh system-random seed.")
    add_device_args(parser)
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
    run_bi_gce_fuzz(args)


if __name__ == "__main__":
    main()
