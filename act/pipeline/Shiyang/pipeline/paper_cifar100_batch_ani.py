"""Reproduction of the paper's Cifar100 Batch-Ani (anisotropic) experiment.

Paper: "Tensor-Based Batch Fuzzing with Adaptive Perturbation Scaling for
Deep Neural Networks" (Zhang & Sui, arXiv:2606.25239), Section 5 / Table 3-4,
Cifar100 row, Batch-Ani column.

This script is a *baseline* reproduction: it deliberately turns OFF every
extension this repo has grown since the paper (HPGD mutation, the
PatternStateManager BK-tree/Bloom-filter admission+scheduling, BI/GCE
two-task loop, BI threading) so the numbers it produces are comparable to
Table 4's Batch-Ani row. Once this baseline is confirmed, the HPGD/BK-tree
state-management extensions (act/config/pipeline.yaml's admission_mode=
"state" / scheduling_mode="sparse" / enable_bi_gce=True path) are meant to
be layered back on top of this exact configuration as an ablation, not
folded in here.

What Table 3/4 actually specify for Cifar100:
  - Benchmark: VNN-COMP cifar100_2024 (2 ONNX models: ResNet-medium,
    ResNet-large; 3x32x32 input; G=2 model groups; B=99/100 specs/group).
  - Batch-Ani = anisotropic adaptive perturbation scaling
    (perturb_mode="adaptive_perdim" in this codebase's terms; Eq. 13:
    S_{b,d} = s * (u_{b,d} - l_{b,d})).
  - Mutation portfolio weights (Table 2): Gradient 0.5, Boundary 0.2,
    Random 0.3. This codebase implements the paper's single "Gradient"
    category (FGSM as the T=1 special case of PGD, Eq. 10-11) as two
    separate MutationEngine strategies, "gradient" (FGSM) and "pgd"
    (iterative); the paper's main-result runs use the iterative PGD
    variant, so weight 0.5 goes entirely to "pgd" here (gradient=0),
    matching act/pipeline/Shiyang/pipeline/batch_aniso.py's
    BATCH_ANISO_WEIGHTS precedent.
  - Coverage: GlobalCov, activation threshold tau=0.1 (Section 5,
    "Experimental Settings").
  - Energy constants alpha=10, beta=100, e_min=0.1 -- already hardcoded in
    ACTFuzzer/MutationEngine (act/pipeline/fuzzing/actfuzzer.py), not
    reconfigurable here; listed for reference only.
  - Per-model-group timeout t_max=60s (NOT a shared 60s budget across
    groups -- Table 4 processes each spec campaign in "a single 60s
    window" per group).
  - Scale factor s=0.1 (README's documented default; also the middle of
    Table 6's RQ3 sweep {0.01,0.05,0.1,0.2,0.3,0.5}).

Usage:
    python -m act.pipeline.Shiyang.pipeline.paper_cifar100_batch_ani
    python -m act.pipeline.Shiyang.pipeline.paper_cifar100_batch_ani \\
        --timeout 60 --perturb-scale 0.1 --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from act.front_end.model_synthesis import synthesize_models_from_specs
from act.front_end.vnnlib_loader.create_specs import VNNLibSpecCreator
from act.pipeline.fuzzing.actfuzzer import ACTFuzzer, FuzzingConfig, FuzzingReport
from act.pipeline.Shiyang.pipeline.common import initial_seeds_from_wrapped_model
from act.util.cli_utils import initialize_from_args

SHIYANG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = SHIYANG_ROOT / "results" / "paper_cifar100_batch_ani"

# Table 2: Gradient 0.5 / Boundary 0.2 / Random 0.3. "Gradient" -> this
# codebase's iterative "pgd" strategy (see module docstring). This is the
# paper-exact baseline portfolio -- hpgd/hpgd_cov both default to 0.0 here so
# reproducing the paper's numbers doesn't silently drift; pass
# --hpgd-cov-weight to add the coverage-targeted extension into the mix for
# an ablation run (not part of the paper).
PAPER_BATCH_ANI_WEIGHTS = {
    "gradient": 0.0,
    "pgd": 0.5,
    "activation": 0.0,
    "boundary": 0.2,
    "random": 0.3,
    "hpgd": 0.0,
    "hpgd_cov": 0.0,
}


def build_mutation_weights(
    hpgd_cov_weight: float, hpgd_weight: float = 0.0
) -> dict[str, float]:
    """Portfolio weights for one run, with the guided strategy DISPLACING (not diluting).

    Exactly one of hpgd_cov_weight / hpgd_weight may be > 0; they occupy the
    same slot in the portfolio and pairing both at once has no matched
    interpretation. Which one to use is not a free choice -- it has to match
    --admission-mode, because the two strategies optimize in DIFFERENT neuron
    spaces and only one of them lines up with each admission rule:

        hpgd_cov + admission "coverage"  -- aligned. The strategy chases
            CoverageTracker's uncovered neurons (Conv2d/Linear/ReLU OUTPUTS,
            channel abs-max-pooled) and admission rewards exactly that signal.
        hpgd + admission "state"  -- aligned. The strategy flips ReLU sign
            patterns restricted to PatternStateManager.candidate_indices (the
            unstable subspace, wired in ACTFuzzer._init_state_manager), and
            admission judges novelty over that same restricted subspace
            (PatternStateManager.pattern_restrict).

    Crossing them decouples the run: hpgd_cov + "state" makes the strategy
    hunt in CoverageTracker's space while admission scores in the unstable
    ReLU-pre-activation subspace, so a sample that lights up a genuinely new
    neuron is still rejected unless its sign pattern also happens to be novel.
    cov_interesting is still computed every iteration and then discarded
    (actfuzzer.py step 6 takes the "state" branch). That crossed pairing is
    what the verify_large_trials_v2 campaign's state_v2 arm actually ran.

    MutationEngine normalizes whatever dict it gets (mutations.py:1371-1374)
    and then samples ONE strategy per iteration from that distribution
    (mutations.py:1525-1527). So simply writing `weights["hpgd_cov"] = w` into
    the baseline dict -- which is what this script did through the
    `verify_large_trials_v2` campaign -- moves the denominator from 1.0 to
    1.0+w and shrinks EVERY baseline entry by 1/(1+w), pgd included: at w=0.5
    pgd's share fell 50% -> 33.3%. hpgd_cov's objective is purely coverage
    (HPGDCoverageMutation carries no CE-margin term at all), so those diverted
    iterations contribute nothing to violation-finding, and the resulting drop
    in counterexample yield can't be separated from the loss of pgd budget.

    So instead: hold pgd at its baseline share, give hpgd_cov the SAME share
    it had under the old additive scheme -- w/(S+w), i.e. 33.3% at w=0.5, so
    its absolute effort stays matched to the archived v2 runs -- and take the
    whole difference out of the remaining non-pgd strategies, rescaled among
    themselves in their baseline proportions.

        strategy    baseline    w=0.5 (here)    w=0.5 (old additive)
        pgd            50.0%          50.0%                   33.3%
        hpgd_cov        0.0%          33.3%                   33.3%
        boundary       20.0%           6.7%                   13.3%
        random         30.0%          10.0%                   20.0%

    Note this CHANGES what --hpgd-cov-weight 0.5 means: re-running the
    archived runners (verify_many_short.sh, verify_many_short_v2.sh,
    verify_long30m.sh, ...) no longer reproduces their recorded numbers.
    """
    if hpgd_cov_weight > 0 and hpgd_weight > 0:
        raise ValueError(
            "Pass --hpgd-cov-weight OR --hpgd-weight, not both: they take the "
            "same displaced share of the portfolio, and each is aligned with a "
            "different --admission-mode, so running them together has no "
            "matched interpretation."
        )

    weights = dict(PAPER_BATCH_ANI_WEIGHTS)
    guided = "hpgd_cov" if hpgd_cov_weight > 0 else "hpgd"
    w_guided = max(hpgd_cov_weight, hpgd_weight)
    if w_guided <= 0:
        return weights

    total = sum(weights.values())
    p_pgd = weights["pgd"] / total
    # Same share the additive scheme gave it, so the guided strategy's own
    # effort is unchanged vs. the v2 campaign and pgd's budget is the only
    # moved variable. Both guided strategies get the identical share, so the
    # cov and state arms are matched on everything except which one runs.
    p_guided = w_guided / (total + w_guided)

    others = {k: v for k, v in weights.items()
              if k not in ("pgd", "hpgd_cov", "hpgd") and v > 0}
    others_mass = sum(others.values())
    remaining = 1.0 - p_pgd - p_guided
    if remaining < 0 or others_mass <= 0:
        raise ValueError(
            f"weight {w_guided} for {guided!r} needs {p_guided:.1%} of the "
            f"portfolio, which cannot be displaced from the non-pgd strategies "
            f"({others_mass / total:.1%} available) without cutting into pgd's "
            f"{p_pgd:.1%}. Lower the weight (max is {total:.3g})."
        )

    weights["pgd"] = p_pgd
    weights[guided] = p_guided
    for name, w in others.items():
        weights[name] = w / others_mass * remaining
    return weights


@dataclass
class GroupResult:
    model_id: str
    batch_size: int
    violations: int
    ttfv_seconds: float | None
    neuron_coverage: float
    total_mutations: int
    total_time: float
    throughput: float


def run_group(
    model_id: Any,
    wrapped_model,
    args: argparse.Namespace,
    group_output_dir: Path,
) -> GroupResult:
    initial_seeds = initial_seeds_from_wrapped_model(wrapped_model)
    if not initial_seeds:
        raise RuntimeError(f"No initial seeds extracted for model group {model_id!r}")

    group_output_dir.mkdir(parents=True, exist_ok=True)

    weights = build_mutation_weights(args.hpgd_cov_weight, args.hpgd_weight)
    if weights.get("hpgd", 0) > 0 and args.admission_mode != "state":
        raise ValueError(
            "--hpgd-weight requires --admission-mode state. HPGDMutation's "
            "flip candidates are restricted to PatternStateManager's unstable "
            "subspace, wired only in ACTFuzzer._init_state_manager; without it "
            "candidate_indices stays None (every neuron in the network is a "
            "flip candidate, intractable on a CNN) and flip_count/num_steps/"
            "margin silently keep their constructor defaults instead of the "
            "configured hpgd_* values."
        )

    config = FuzzingConfig.from_yaml(
        max_iterations=args.max_iterations,
        timeout_seconds=args.timeout,
        seed_selection_strategy="energy",
        mutation_weights=weights,
        coverage_strategy="GlobalCov",
        activation_threshold=0.1,
        perturb_mode="adaptive_perdim",
        perturb_scale=args.perturb_scale,
        save_counterexamples=args.save_counterexamples,
        output_dir=group_output_dir,
        report_interval=args.report_interval,
        verbose=args.verbose,
        trace_level=0,
        trace_sample_rate=1,
        trace_storage="json",
        trace_output=None,
        stop_on_first_violation=False,
        # Default "coverage": paper baseline, not the BK-tree/HPGD state-
        # management extension this repo has since grown on top of it.
        # --admission-mode state / --scheduling-mode sparse opt into that
        # extension as an ablation (not part of the paper reproduction).
        admission_mode=args.admission_mode,
        scheduling_mode=args.scheduling_mode,
        enable_bi_gce=False,
        bi_threaded=False,
        # ACTFuzzer._fuzz_iteration branches on bi_attack_strategy ALONE
        # (independent of enable_bi_gce/admission_mode/scheduling_mode --
        # see actfuzzer.py: "if self.config.bi_attack_strategy is not None:
        # inputs = self._mutate_hpgd_then_pgd(...) else: inputs =
        # self.mutation_engine.mutate(seeds)"). Leaving this unset would
        # silently inherit whatever act/config/pipeline.yaml currently has
        # on disk (e.g. "apgd_dlr" during BI/GCE development) and bypass
        # `mutation_weights` entirely, regardless of every other override
        # above. Must be forced to None for this to actually be the paper's
        # weighted-portfolio dispatch.
        bi_attack_strategy=None,
        hpgd_cov_nearest_margin=args.hpgd_cov_nearest_margin,
    )

    print(f"\n[paper_cifar100_batch_ani] Fuzzing model group {model_id!r} "
          f"(B={len(initial_seeds)}, t_max={args.timeout}s)")
    print("[paper_cifar100_batch_ani] Mutation portfolio: "
          + ", ".join(f"{k}={v / sum(weights.values()):.1%}"
                      for k, v in weights.items() if v > 0))

    fuzzer = ACTFuzzer(
        wrapped_model=wrapped_model,
        initial_seeds=initial_seeds,
        config=config,
    )
    start_time = time.time()
    report: FuzzingReport = fuzzer.fuzz()

    ce_timestamps = [ce.timestamp for ce in report.counterexamples]
    ttfv = (min(ce_timestamps) - start_time) if ce_timestamps else None
    throughput = report.total_mutations / report.total_time if report.total_time > 0 else 0.0

    result = GroupResult(
        model_id=str(model_id),
        batch_size=len(initial_seeds),
        violations=len(report.counterexamples),
        ttfv_seconds=ttfv,
        neuron_coverage=report.neuron_coverage,
        total_mutations=report.total_mutations,
        total_time=report.total_time,
        throughput=throughput,
    )

    # distinct_instances_hit: which of this group's B seeds (0-based position
    # within initial_seeds, == Counterexample.spec_row == SeedCorpus's
    # original_index) produced at least one counterexample -- lets a caller
    # count how many DISTINCT instances got broken, not just raw CE volume
    # (a single instance can spawn hundreds of near-duplicate CEs once the
    # energy-weighted corpus snowballs on it; see the seed-starvation
    # discussion this was added to answer).
    #
    # Was ce.seed_index until upstream PR #107 rebuilt Counterexample around
    # spec_row/severity/true_class. spec_row carries the same quantity: the
    # OutputSpec row backing the lane, which PropertyChecker now passes as
    # rows=seeds.original_index precisely because a corpus that samples with
    # replacement makes lane i != instance i.
    distinct_instances_hit = sorted({ce.spec_row for ce in report.counterexamples
                                      if ce.spec_row is not None})

    with open(group_output_dir / "group_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **result.__dict__,
                "iterations": report.total_iterations,
                "seeds_explored": report.seeds_explored,
                "distinct_instances_hit": distinct_instances_hit,
            },
            f,
            indent=2,
        )

    print(
        f"[paper_cifar100_batch_ani] group={model_id!r}: "
        f"violations={result.violations} ttfv={result.ttfv_seconds} "
        f"cov={result.neuron_coverage:.4f} thpt={result.throughput:.1f}it/s"
    )
    return result


def print_table4_row(benchmark: str, groups: list[GroupResult]) -> None:
    total_violations = sum(g.violations for g in groups)
    finite_ttfv = [g.ttfv_seconds for g in groups if g.ttfv_seconds is not None]
    ttfv = min(finite_ttfv) if finite_ttfv else float("nan")
    total_mutations = sum(g.total_mutations for g in groups)
    total_time = sum(g.total_time for g in groups)
    throughput = total_mutations / total_time if total_time > 0 else 0.0
    # Weighted by batch size, matching how Table 4 aggregates neuron
    # coverage across G model groups within a benchmark.
    total_b = sum(g.batch_size for g in groups)
    neuron_cov = (
        sum(g.neuron_coverage * g.batch_size for g in groups) / total_b if total_b else 0.0
    )

    print(f"\n{'=' * 78}")
    print(f"Table 4 reproduction -- {benchmark} / Batch-Ani")
    print(f"{'=' * 78}")
    print(f"{'Method':<12}{'Violations':>12}{'TTFV(sec)':>12}{'NeuronCov(%)':>14}{'Thpt(it/s)':>12}")
    print(
        f"{'Batch-Ani':<12}{total_violations:>12}{ttfv:>12.2f}"
        f"{neuron_cov * 100:>14.2f}{throughput:>12.1f}"
    )
    for g in groups:
        cov = g.ttfv_seconds if g.ttfv_seconds is not None else float("nan")
        print(
            f"  - {g.model_id:<40}{g.violations:>10}{cov:>12.2f}"
            f"{g.neuron_coverage * 100:>14.2f}{g.throughput:>12.1f}   (B={g.batch_size})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m act.pipeline.Shiyang.pipeline.paper_cifar100_batch_ani",
        description=(
            "Reproduce the paper's Cifar100 Batch-Ani (anisotropic) fuzzing "
            "experiment (Table 3/4) as a baseline for later HPGD/BK-tree work."
        ),
    )
    parser.add_argument("--category", default="cifar100_2024")
    parser.add_argument(
        "--max-instances", type=int, default=200,
        help="Total VNNLIB instances to load across the whole category (all "
             "model groups combined; cifar100_2024 has 200 on disk). Ignored "
             "when --instance-indices is given.",
    )
    parser.add_argument(
        "--instance-indices", default=None,
        help="Restrict to specific 0-based rows of the category's "
             "instances.csv instead of a 'first --max-instances' prefix -- "
             "e.g. '100-199' or '100,101,105' to load only one model group "
             "(cifar100_2024: rows 0-99=resnet_medium, 100-199=resnet_large) "
             "for faster repeated-trial verification runs. Comma-separated "
             "list of ints and/or 'a-b' inclusive ranges.",
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0,
        help="Per-model-group wall-clock budget in seconds (paper: t_max=60s).",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=10_000_000,
        help="Iteration cap per group (effectively unbounded; --timeout governs stopping).",
    )
    parser.add_argument(
        "--perturb-scale", type=float, default=0.1,
        help="Anisotropic scale factor s (Eq. 13). Paper README/Table 6 default: 0.1.",
    )
    parser.add_argument(
        "--hpgd-cov-weight", type=float, default=0.0,
        help="Not part of the paper baseline (default 0 = off). >0 adds the "
             "coverage-targeted 'hpgd_cov' mutation strategy to the portfolio -- "
             "each sample chases its own randomly-drawn never-activated neuron. "
             "It takes a w/(1+w) share (33.3%% at w=0.5) DISPLACED entirely from "
             "boundary/random; pgd stays at its baseline 50%% so CE-seeking "
             "budget is matched against the baseline. See build_mutation_weights "
             "for the table and for why this differs from the additive scheme "
             "used through the verify_large_trials_v2 campaign. See "
             "act/config/pipeline.yaml's hpgd_cov_* keys for step-count/"
             "momentum/decay tuning.",
    )
    parser.add_argument(
        "--hpgd-weight", type=float, default=0.0,
        help="Not part of the paper baseline (default 0 = off). >0 adds the "
             "pattern-space 'hpgd' mutation strategy, taking the SAME displaced "
             "share --hpgd-cov-weight would (33.3%% at 0.5) so the two are "
             "matched. Requires --admission-mode state, which is what wires its "
             "flip candidates to the unstable subspace that admission also "
             "scores in. Mutually exclusive with --hpgd-cov-weight; see "
             "build_mutation_weights for why each guided strategy pairs with "
             "exactly one admission mode.",
    )
    parser.add_argument(
        "--admission-mode", choices=["coverage", "state"], default="coverage",
        help="Not part of the paper baseline (default 'coverage' = original "
             "GlobalCov-driven admission/energy). 'state' switches ALL "
             "strategies' admission+energy to PatternStateManager's BK-tree "
             "pattern-novelty check instead (act/pipeline/fuzzing/state_manager.py) "
             "-- see FuzzingConfig.admission_mode.",
    )
    parser.add_argument(
        "--scheduling-mode", choices=["energy", "sparse"], default="energy",
        help="Not part of the paper baseline. 'sparse' draws seeds from "
             "PatternStateManager's BK-tree registry instead of SeedCorpus's "
             "energy-weighted select(). See FuzzingConfig.scheduling_mode.",
    )
    parser.add_argument(
        "--hpgd-cov-nearest-margin", action="store_true",
        help="Only meaningful when --hpgd-cov-weight > 0. Instead of "
             "hpgd_cov_target_count random uncovered targets/sample, chase the "
             "single uncovered neuron closest to activation_threshold from below "
             "(easiest to flip) per sample. See FuzzingConfig.hpgd_cov_nearest_margin.",
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Run this many independent trials in ONE process, writing each to "
             "<output>/rep<N>/. The VNNLIB load and model synthesis are paid "
             "once instead of per trial -- on cifar100_2024 that setup is 109s "
             "against 120s of fuzzing, so per-trial re-invocation spends nearly "
             "half a campaign re-reading the same ONNX files. Each repeat still "
             "builds a fresh ACTFuzzer, so corpus/coverage/state start empty.",
    )
    parser.add_argument("--report-interval", type=int, default=2000)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--no-save", action="store_false", dest="save_counterexamples")
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT_DIR.relative_to(SHIYANG_ROOT)),
        help="Output dir, relative to act/pipeline/Shiyang/ unless absolute.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "gpu"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = parser.parse_args()

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = SHIYANG_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    initialize_from_args(args)

    instance_indices = None
    if args.instance_indices:
        instance_indices = []
        for part in args.instance_indices.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                instance_indices.extend(range(int(lo), int(hi) + 1))
            else:
                instance_indices.append(int(part))

    print(f"[paper_cifar100_batch_ani] Loading VNNLIB category={args.category!r} "
          f"(max_instances={args.max_instances}, instance_indices={instance_indices})")
    spec_results = VNNLibSpecCreator().create_specs_for_data_model_pairs(
        categories=[args.category],
        max_instances=args.max_instances,
        instance_indices=instance_indices,
    )
    if not spec_results:
        raise RuntimeError(
            f"No VNNLIB specs were loaded for category={args.category!r}. "
            f"Download it first with: python -m act.pipeline --download {args.category}"
        )

    print(f"[paper_cifar100_batch_ani] Synthesizing wrapped models "
          f"from {len(spec_results)} spec result(s)")
    wrapped_models = synthesize_models_from_specs(spec_results)
    if not wrapped_models:
        raise RuntimeError("ACT model synthesis produced no wrapped models.")
    print(f"[paper_cifar100_batch_ani] Got G={len(wrapped_models)} model group(s)")

    # --repeat runs the trials inside one process so the ONNX load and model
    # synthesis above are paid once instead of per trial. On cifar100_2024 that
    # setup is 109s against 120s of actual fuzzing, so re-invoking the module
    # per trial spends nearly half the campaign re-reading the same files.
    # Each repeat still builds a fresh ACTFuzzer (and so a fresh corpus,
    # coverage tracker and state manager) inside run_group, so trials stay
    # independent; only the immutable loaded models are shared.
    for rep in range(1, args.repeat + 1):
        rep_dir = output_dir if args.repeat == 1 else output_dir / f"rep{rep}"
        if args.repeat > 1:
            print(f"\n{'#' * 78}\n# repeat {rep}/{args.repeat}\n{'#' * 78}")
        group_results: list[GroupResult] = []
        for model_id, wrapped_model in wrapped_models.items():
            safe_name = "_".join(map(str, model_id)) if isinstance(model_id, tuple) else str(model_id)
            group_dir = rep_dir / safe_name.replace("/", "_").replace("\\", "_")
            group_results.append(run_group(model_id, wrapped_model, args, group_dir))

        print_table4_row(args.category, group_results)

        with open(rep_dir / "table4_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "benchmark": args.category,
                    "method": "Batch-Ani",
                    "perturb_scale": args.perturb_scale,
                    "timeout_per_group_seconds": args.timeout,
                    "groups": [g.__dict__ for g in group_results],
                    "total_violations": sum(g.violations for g in group_results),
                },
                f,
                indent=2,
            )
        print(f"\n[paper_cifar100_batch_ani] Full summary written to "
              f"{rep_dir / 'table4_summary.json'}")


if __name__ == "__main__":
    main()
