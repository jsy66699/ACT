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
import hashlib
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
    hpgd_cov_weight: float, hpgd_weight: float = 0.0, absorb_non_pgd: bool = False,
    pgd_only: bool = False,
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
    if pgd_only:
        # The portfolio ablated away entirely. Its own branch, so no other flag
        # combination's weights move: everything below is untouched.
        if hpgd_weight > 0 or hpgd_cov_weight > 0:
            raise ValueError("--pgd-only takes the whole portfolio; it cannot be "
                             "combined with a guided-strategy weight.")
        return {"pgd": 1.0}

    if absorb_non_pgd:
        # A separate, explicit branch rather than a tweak to the displacement
        # arithmetic below: everything that is not pgd becomes hpgd, so the
        # portfolio is exactly {pgd at its baseline share, hpgd with all the
        # rest}. Boundary and random are dropped entirely. Every other flag
        # combination still routes through the untouched logic below, so no
        # existing arm's weights move.
        if hpgd_weight <= 0:
            raise ValueError(
                "absorb_non_pgd needs --hpgd-weight > 0: there is no hpgd "
                "strategy to give the displaced share to."
            )
        if hpgd_cov_weight > 0:
            raise ValueError("absorb_non_pgd is for the hpgd portfolio, not hpgd_cov.")
        base = dict(PAPER_BATCH_ANI_WEIGHTS)
        total = sum(base.values())
        p_pgd = base["pgd"] / total
        return {"pgd": p_pgd, "hpgd": 1.0 - p_pgd}

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


def _dump_founder_clusters(fuzzer, out_dir: Path, cap: int) -> None:
    """Sample counterexamples from EVERY instance that produced any.

    Diversity is a two-level quantity -- how many instances were broken, and how
    varied one instance's counterexamples are -- so a dump of the single richest
    instance can only ever answer the second half. Every instance with
    counterexamples is sampled here, `cap` of them each, and instances are kept
    separable by an index array rather than by separate files.

    Sampling is uniform over the instance's counterexample ROWS, not over its
    lineages: weighting by lineage would presuppose the founder partition, which
    is one of the things the later analysis has to check rather than assume.

    Stored as float16. The consumer differentiates the margin at these points to
    recover each one's affine constraint, and that gradient is stable to far
    less precision than float32 carries; full precision here would multiply the
    file size for no change in any downstream number.
    """
    import numpy as np
    by_inst = fuzzer.seed_corpus.ce_founder_rows()
    if not by_inst:
        return
    spec = fuzzer.output_spec
    y = getattr(spec, "y_true", None)
    if y is not None:
        y = y.reshape(-1)

    rng = np.random.default_rng(0)
    xs, inst_ids, founder_ids, true_cls = [], [], [], []
    for inst, lineages in sorted(by_inst.items()):
        rows, founders = [], []
        for f, rs in lineages.items():
            rows.extend(rs)
            founders.extend([f] * len(rs))
        rows = np.asarray(rows)
        founders = np.asarray(founders)
        if len(rows) > cap:
            pick = rng.choice(len(rows), cap, replace=False)
            rows, founders = rows[pick], founders[pick]
        xs.append(fuzzer.seed_corpus.row_tensors(rows.tolist())
                  .detach().cpu().numpy().reshape(len(rows), -1).astype(np.float16))
        inst_ids.append(np.full(len(rows), inst, np.int32))
        founder_ids.append(founders.astype(np.int64))
        if y is None:
            true_cls.append(-1)
        else:
            true_cls.append(int(y[inst]) if y.numel() > 1 else int(y[0]))

    np.savez_compressed(
        out_dir / "ce_sample.npz",
        x=np.concatenate(xs),
        instance=np.concatenate(inst_ids),
        founder=np.concatenate(founder_ids),
        instances=np.array(sorted(by_inst)),
        true_class=np.array(true_cls),
        # Population size per instance, before the cap -- the rarefaction step
        # needs to know what was sampled FROM, not just what came out.
        ce_total=np.array([sum(len(r) for r in by_inst[i].values())
                           for i in sorted(by_inst)]),
    )


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

    weights = build_mutation_weights(args.hpgd_cov_weight, args.hpgd_weight,
                                     absorb_non_pgd=args.hpgd_absorb_non_pgd,
                                     pgd_only=args.pgd_only)
    if args.hpgd_flip_frac is not None and weights.get("hpgd", 0) <= 0:
        raise ValueError(
            "--hpgd-flip-frac only affects HPGDMutation, which is not in the "
            "portfolio unless --hpgd-weight > 0. Passing it alone silently "
            "changes nothing, so it is rejected rather than ignored."
        )
    if ((args.hpgd_target_mode != "random_flip" or args.hpgd_schedule != "off")
            and weights.get("hpgd", 0) <= 0):
        raise ValueError(
            "--hpgd-target-mode / --hpgd-schedule only affect HPGDMutation, "
            "which is not in the portfolio unless --hpgd-weight > 0. Passing "
            "them alone silently changes nothing, so they are rejected rather "
            "than ignored."
        )
    if weights.get("hpgd", 0) > 0 and args.admission_mode not in ("state", "state_always"):
        raise ValueError(
            "--hpgd-weight requires --admission-mode state/state_always. HPGDMutation's "
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
        unstable_mask_source=args.unstable_mask_source,
        unstable_mask_grad_threshold=args.unstable_mask_grad_threshold,
        state_bins=args.state_bins,
        state_bin_tau=args.state_bin_tau,
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
        hpgd_flip_frac=args.hpgd_flip_frac,
        unstable_mask_scope=args.unstable_mask,
        state_dims=args.state_dims,
        state_dim_select=args.state_dim_select,
        ce_prior_path=args.ce_prior or "",
        # Instance indices are per model group, so the prior has to be looked
        # up under the group it was harvested from.
        ce_prior_key=str(model_id),
        hpgd_target_mode=args.hpgd_target_mode,
        hpgd_schedule=args.hpgd_schedule,
        hpgd_expand_frac=args.hpgd_expand_frac,
        hpgd_expand_patience=args.hpgd_expand_patience,
        coverage_per_instance=args.coverage_per_instance,
        ce_energy_bonus=args.ce_energy,
        energy_tiers=args.energy_tiers or "",
        ce_parent_replacement=args.ce_parent_replacement,
        select_with_replacement=not args.select_without_replacement,
        select_per_instance=args.select_per_instance,
        ce_parent_energy_threshold=args.ce_parent_energy_threshold,
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
    if args.dump_ce_patterns:
        n = fuzzer.dump_ce_patterns(args.dump_ce_patterns, str(model_id))
        print(f"[paper_cifar100_batch_ani] dumped raw CE patterns for {n} instance(s)")
    if args.dump_ce_prior:
        n = fuzzer.dump_ce_prior(args.dump_ce_prior, str(model_id))
        print(f"[paper_cifar100_batch_ani] harvested a CE prior for {n} instance(s) "
              f"-> {args.dump_ce_prior}")

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

    if args.dump_founder_clusters:
        _dump_founder_clusters(fuzzer, group_output_dir, args.dump_founder_clusters)

    with open(group_output_dir / "group_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                **result.__dict__,
                "iterations": report.total_iterations,
                "seeds_explored": report.seeds_explored,
                "distinct_instances_hit": distinct_instances_hit,
                # The A/B evidence for --ce-parent-replacement: rows is every
                # seed ever admitted, live is what select() could still draw
                # at the end, retired is how many CE parents were replaced.
                # retired == 0 on every arm that leaves the flag off.
                "corpus_rows": fuzzer.seed_corpus.rows,
                "corpus_live": len(fuzzer.seed_corpus),
                "corpus_retired": fuzzer.seed_corpus.retired,
                # What share of select()'s draw the CE tier still owns at the
                # end -- the starvation itself, rather than its symptom.
                # Flag-based, not threshold-based: with --ce-energy below the
                # admitted bonus of 10 the CE tier sits UNDER an ordinary novel
                # seed, and every threshold reading came back 0. Identical to
                # the old energy_mass_above(100) at the shipped ce_energy, so
                # the column stays comparable with every arm run before this.
                "corpus_ce_energy_mass": fuzzer.seed_corpus.ce_mass(),
                # Count, not mass: how much of the POOL the CE tier occupies.
                # A tier can own a fifth of the draw while being a fortieth of
                # the pool, so the two answer different saturation questions.
                "corpus_ce_count": fuzzer.seed_corpus.ce_count(),
                # Why children did not become rows, split three ways. The
                # single rejection rate reported before sums the admission
                # gate, hash dedup, and cerepl's one-slot-per-parent rule,
                # and the last of those exists only on cerepl arms -- which
                # is why that rate was never comparable across arms.
                "corpus_drops": fuzzer.seed_corpus.drop_stats,
                # Draws per instance lineage, indexed by original_index. The
                # scheduling argument is entirely about this vector's shape:
                # an instance drawn zero times cannot be broken no matter how
                # long the run. B numbers, so cheap enough to keep per group.
                "corpus_draws_by_instance": fuzzer.seed_corpus.draws_by_instance(),
                # Per instance, how many DISTINCT parents its counterexamples
                # came from. Separates real exploration from one lineage being
                # re-drawn: violations cannot tell 5000 finds from 5000 children
                # of one seed. Covers retired rows too, so cerepl arms are not
                # undercounted.
                "corpus_ce_lineage": fuzzer.seed_corpus.ce_lineage_by_instance(),
                # The novelty predicate's own verdict, separated from the gate:
                # observed / admitted / ce / ce_and_novel. Empty under
                # --admission-mode coverage, which never runs the predicate.
                "state_novelty": getattr(fuzzer, "_novelty", {}),
                # Same four counts split by the strategy that produced the
                # batch, because a change to one strategy's targeting is
                # otherwise diluted by that strategy's portfolio share.
                "state_novelty_by_strategy": getattr(fuzzer, "_novelty_by_strategy", {}),
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
        "--dump-founder-clusters", type=int, default=0, metavar="N",
        help="Per group, write founder_clusters.npz: the input tensor of each "
             "of up to N founders for the instance with the most of them, plus "
             "one sibling per founder. Off (0) by default. Lets a later pass "
             "ask whether separate breakthroughs landed in separate activation "
             "regions or rediscovered the same one.",
    )
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
        "--admission-mode", choices=["coverage", "state", "always", "state_always"],
        default="coverage",
        help="Not part of the paper baseline (default 'coverage' = original "
             "GlobalCov-driven admission/energy). 'state' switches ALL "
             "strategies' admission+energy to PatternStateManager's BK-tree "
             "pattern-novelty check instead (act/pipeline/fuzzing/state_manager.py). "
             "'always' opens COVERAGE's gate to 100%% while keeping coverage's "
             "energy formula; 'state_always' opens the gate to 100%% while "
             "keeping STATE's energy formula, which is the arm that separates "
             "what admission lets in from what it makes the sample worth "
             "-- see FuzzingConfig.admission_mode.",
    )
    parser.add_argument(
        "--state-bins", type=int, choices=[2, 3], default=2,
        help="How a pre-activation becomes a state coordinate. 2 (default) is "
             "sign(z) -- the ReLU partition, one bit per neuron. 3 splits at "
             "z = -tau, +tau instead (two walls, three segments), encoded as "
             "two bits per neuron so the BK-tree/Bloom/marginals are unchanged. "
             "On a smooth activation almost no neuron can reach the sign wall "
             "inside its box while ~2.3x as many can reach a +-tau wall, which "
             "is the reason this exists; on ReLU the sign wall IS the kink and "
             "3 has no such motivation. Applies to admission, the state energy "
             "formula and HPGD's hinge target alike.",
    )
    parser.add_argument(
        "--state-bin-tau", type=float, default=1.0,
        help="Wall position for --state-bins 3. Fixed, not per-neuron: on "
             "layers whose |z| runs to 20 a fixed tau leaves most neurons with "
             "all three segments collapsed into one, which is a real limitation "
             "of this first version.",
    )
    parser.add_argument(
        "--scheduling-mode", choices=["energy", "sparse"], default="energy",
        help="Not part of the paper baseline. 'sparse' draws seeds from "
             "PatternStateManager's BK-tree registry instead of SeedCorpus's "
             "energy-weighted select(). See FuzzingConfig.scheduling_mode.",
    )
    parser.add_argument(
        "--hpgd-flip-frac", type=float, default=None,
        help="Size HPGD's flip budget as a fraction of the unstable candidate "
             "set instead of the fixed hpgd_flip_count. The fixed count is not "
             "comparable across networks -- 10 flips is 10.8%% of safenlp's 93 "
             "unstable neurons but 0.31%% of a cifar100 ResNet's 3225 -- so "
             "matching this fraction is what makes a cross-benchmark HPGD "
             "comparison a matched one. Requires --admission-mode state.",
    )
    parser.add_argument(
        "--state-dims", type=float, default=0.0,
        help="Thin each instance's state vector to this many dimensions "
             "(0 = all; <1 = a fraction of that instance's own unstable set). "
             "Needs --unstable-mask per_instance. Admission asks 'seen this "
             "state before', which only informs when the space is coarse "
             "enough to revisit: an instance's full set is ~883 dims against "
             "~4000 samples a trial, so nothing repeats and everything is "
             "admitted. The shipped row0 mask cut it to ~116 varying dims by "
             "accident and did reject 20-27%%.",
    )
    parser.add_argument(
        "--dump-ce-prior", default=None,
        help="After each group, append per-instance counterexample suspiciousness "
             "(|P(sign=+|CE) - P(sign=+|rest)|) to this .pt file. Merges across "
             "runs and groups, so several harvest runs sharpen the same prior.",
    )
    parser.add_argument(
        "--dump-ce-patterns", default=None,
        help="Dump the raw per-instance sign patterns (counterexample and not) "
             "to this .pt, for analysing the STRUCTURE of an instance's "
             "counterexamples -- how many distinct linear regions they occupy, "
             "how far apart they are, which neurons are constant across them. "
             "The prior dump only keeps per-neuron marginals, which cannot "
             "answer any of that.",
    )
    parser.add_argument(
        "--ce-prior", default=None,
        help="Read state-dimension scores from a file written by --dump-ce-prior. "
             "Only used with --state-dim-select ce_prior. This is an ORACLE: the "
             "scores come from counterexamples a previous run already found, so it "
             "measures whether that information is worth anything, not a technique "
             "that could run online.",
    )
    parser.add_argument(
        "--state-dim-select",
        choices=["random", "flip_freq", "flip_rare", "margin_grad", "ce_prior"],
        default="random",
        help="Which dimensions to keep. 'random' is the control that isolates "
             "the dimension COUNT from the choice of dimensions. 'flip_freq' "
             "keeps the neurons whose sign is most balanced over random points "
             "of the instance's own box -- the ones that carry entropy, as "
             "opposed to the ones interval propagation merely cannot rule out. "
             "'flip_rare' keeps the least balanced ones that still move at all: "
             "novelty admission only filters when coordinates COLLIDE, and "
             "rejection was measured to track coordinate movement monotonically "
             "(22.7%% row0 / 9.9%% random / 3.5%% flip_freq), so stillness is "
             "what to select for.",
    )
    parser.add_argument(
        "--unstable-mask", choices=["row0", "union", "per_instance"], default="row0",
        help="Which spec rows the unstable-neuron mask comes from. 'row0' "
             "(default, and what every earlier arm ran) derives it from the "
             "FIRST instance's box alone and applies it to all B lanes -- "
             "measured on cifar100_2024, a lane's own unstable set overlaps it "
             "by only 9-12%%, and just 6.6-7.7%% of the sign flips that "
             "actually occur land inside it. 'union' ORs every row's box, "
             "giving a true superset of each lane's own set -- but a lane's "
             "own neurons are then only ~5%% of it, so flip targets drawn from "
             "it miss and novelty measured over it is diluted to always-novel. "
             "'per_instance' keeps that union as the axis and additionally "
             "masks each lane to its OWN unstable set when choosing flips and "
             "when fingerprinting, which is the only setting where the "
             "guidance both reaches and aims. Costs one bound-propagation pass "
             "per instance, cached across repeats. The mask exists only under "
             "--admission-mode state, and there it scopes BOTH HPGD's flip "
             "candidates AND the novelty fingerprint admission itself is keyed "
             "on -- an arm with no hpgd in its portfolio still moves with it.",
    )
    parser.add_argument(
        "--hpgd-target-mode", choices=["random_flip", "interp_real"],
        default="random_flip",
        help="Only meaningful with --hpgd-weight > 0. 'random_flip' (default, "
             "unchanged) flips k bits of the seed's own pattern independently, "
             "which names a sign assignment nothing guarantees is satisfiable: "
             "measured on cifar100_2024, ~15%% of the asked flips land, 0%% "
             "exactly, and the projection ends FARTHER from its own target than "
             "it started. 'interp_real' aims at a cell BETWEEN the seed and one "
             "of the same instance's real recorded states -- feasible by "
             "construction, filtered to unvisited -- reached exactly ~95-97%%. "
             "See FuzzingConfig.hpgd_target_mode.",
    )
    parser.add_argument(
        "--pgd-only", action="store_true",
        help="Portfolio becomes {pgd 1.0}: boundary, random and any guided "
             "strategy are dropped. The ablation that asks whether the "
             "portfolio contributes anything at all, given that every guided "
             "strategy measured here has been at or below the arm without it.",
    )
    parser.add_argument(
        "--hpgd-absorb-non-pgd", action="store_true",
        help="Portfolio becomes exactly {pgd 50%%, hpgd 50%%}: boundary and "
             "random are dropped and their share goes to hpgd, while pgd keeps "
             "its baseline share. Requires --hpgd-weight > 0. Not a variant of "
             "the displacement rule -- it is a separate branch, so every other "
             "flag combination keeps the weights it has today. Note this makes "
             "the arm NOT comparable to statehpgd*, which runs the 4-strategy "
             "portfolio: it needs its own no-schedule control.",
    )
    parser.add_argument(
        "--hpgd-schedule", choices=["off", "coarse_to_fine"], default="off",
        help="Only meaningful with --hpgd-weight > 0. Gives each INSTANCE a "
             "two-phase curriculum instead of one fixed targeting rule: COARSE "
             "large random flips while that instance's state frontier is still "
             "growing (imprecise, but the displacement is what pushes the "
             "boundary out and stocks the registry with distant real states), "
             "then FINE interpolated targets once it stops growing (reached "
             "exactly ~94-99%%), with rho annealed down so precision rises. "
             "Per instance and reversible. See FuzzingConfig.hpgd_schedule.",
    )
    parser.add_argument(
        "--hpgd-expand-frac", type=float, default=0.05,
        help="Coarse-phase flip budget as a share of the candidate set "
             "(default 5%%: ~48 of medium's 951, ~161 of large's 3225, against "
             "hpgd_flip_count=10 in the fine phase).",
    )
    parser.add_argument(
        "--hpgd-expand-patience", type=int, default=20,
        help="Admissions an instance may make without growing its frontier "
             "radius before its lanes switch to the fine phase.",
    )
    parser.add_argument(
        "--coverage-per-instance", action="store_true",
        help="Give each verification instance its own GlobalCov 'already "
             "covered' set instead of one union over the batch. Under the "
             "union (default, and what every earlier arm ran) a sample counts "
             "as interesting only if it fires a neuron NO instance has ever "
             "fired, so instance A raises the bar for the other 99 and the bar "
             "keeps rising -- a harsher admission rule purely for being run in "
             "a bigger batch. Identical to the default at B=1. Reported neuron "
             "coverage stays the union, so only admission changes. See "
             "FuzzingConfig.coverage_per_instance.",
    )
    parser.add_argument(
        "--energy-tiers", default=None,
        help="Four explicit seed energies 'plain,admitted,ce,ce_and_admitted', "
             "e.g. '1,5,10,15'. Replaces the shipped additive formula, whose "
             "tiers are {0.1, 10, 100, 110} -- a counterexample outweighing a "
             "plain seed 1000:1. Sets the RATIOS directly instead of scaling "
             "one term.",
    )
    parser.add_argument(
        "--ce-energy", type=float, default=100.0,
        help="Energy a counterexample seed carries in the corpus, against an "
             "admitted seed's 10. The shipped 100 makes the CE tier own the "
             "draw (measured on safenlp: 100%% of the live corpus and of the "
             "sampling mass). Lowering it raises every non-CE seed's pick "
             "probability. Pass --ce-parent-energy-threshold at or below this "
             "value if also using --ce-parent-replacement.",
    )
    parser.add_argument(
        "--select-per-instance", action="store_true",
        help="Not part of the paper baseline (default off). Round-robin over INSTANCES, energy-weighted only within each, so every instance contributes a lane before any instance contributes a second. This is the instance-level fix that --select-without-replacement is not: that one draws distinct ROWS, and a dominant instance owns thousands of rows, so measured on mnist it left the allocation untouched (Gini 0.77 -> 0.76, Simpson effective instances 4.7 -> 4.7, against 30 instances in the group). With B equal to the instance count -- mnist's 30 specs on 30 lanes -- every batch is a permutation and starvation is impossible by construction. Overrides --select-without-replacement.",
    )
    parser.add_argument(
        "--select-without-replacement", action="store_true",
        help="Not part of the paper baseline (default off = the paper's draw). "
             "The paper draws the B lanes WITH replacement (Algorithm 3 line 10, "
             "'high-energy seeds may repeat'), which is standard exploitation "
             "when the corpus serves one program under test. At B>1 the same "
             "draw is allocating budget across B independent verification "
             "problems, and repetition becomes winner-take-all: measured on "
             "tinyimagenet at B=200, about 12 counterexample seeds took ~161 of "
             "the 200 lanes while the 200 per-instance initial seeds shared "
             "0.01%% of the sampling mass. Even the first batch, with every "
             "energy still 1.0, reaches only B(1-1/e) = 63%% of the instances. "
             "This flag draws B DISTINCT rows instead, still energy-weighted, "
             "so the first batch is a permutation and no lineage can hold two "
             "lanes. Falls back to replacement whenever the live pool is "
             "smaller than B (CE-parent replacement can prune it that far).",
    )
    parser.add_argument(
        "--ce-parent-replacement", action="store_true",
        help="Not part of the paper baseline (default off). Stops the "
             "counterexample tier from monopolising seed selection: a CE child "
             "whose parent is already in that tier takes the parent's corpus "
             "slot instead of being appended beside it (one for one per "
             "parent). Seeds below the tier -- new and still-uncracked "
             "instances -- are never retired. Measured motivation: the CE tier "
             "held 95%% of select()'s probability mass from 3 instances, and a "
             "single 60s trial reached 6.4 distinct instances where the union "
             "over 30 trials of the same arm reached 23. See "
             "FuzzingConfig.ce_parent_replacement.",
    )
    parser.add_argument(
        "--ce-parent-energy-threshold", type=float, default=100.0,
        help="Only meaningful with --ce-parent-replacement. Energy at or above "
             "which a parent counts as already in the CE tier. Energies take "
             "only four values ({0.1, 10, 100, 110}), so 100 (default) is the "
             "whole CE tier and 110 narrows it to counterexamples that were "
             "also pattern-novel.",
    )
    parser.add_argument(
        "--hpgd-cov-nearest-margin", action="store_true",
        help="Only meaningful when --hpgd-cov-weight > 0. Picks the SAME "
             "hpgd_cov_target_count targets/sample, but chooses the ones closest "
             "to activation_threshold from below (easiest to fire) instead of "
             "drawing them uniformly from the whole uncovered pool. Target count "
             "is held fixed so the two rules differ only in HOW targets are "
             "chosen. See FuzzingConfig.hpgd_cov_nearest_margin.",
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
    parser.add_argument(
        "--batch-conversion", action="store_true",
        help="Pin each ONNX's SYMBOLIC batch dimension to the number of "
             "instances sharing it, instead of to 1. Only graphs whose shapes "
             "depend on the batch need it -- attention graphs, which concat a "
             "CLS token and Reshape by a computed shape, otherwise convert to "
             "a module usable at B=1 alone and the whole group cannot be "
             "batched (vit_2023 fails with 'size of tensor a (401) must match "
             "tensor b (5)'). Off by default: for every other benchmark the "
             "conversion is identical either way.",
    )
    parser.add_argument(
        "--unstable-mask-source", choices=["ibp", "gradient"], default="ibp",
        help="How the candidate set is built. 'ibp' (default) propagates "
             "intervals through the back_end. 'gradient' keeps a coordinate "
             "when the box's first-order budget eps*||dz/dx||_1 reaches the "
             "wall it must cross -- not sound, but interval propagation's "
             "error MULTIPLIES through attention's variable-times-variable "
             "products, so on vit_2023 it returns 960/960 while this returns "
             "74/960 (and 24/600 vs 355/600 on the ERAN sigmoid MLP). Costs "
             "one backward per neuron, once per group.",
    )
    parser.add_argument(
        "--unstable-mask-grad-threshold", type=float, default=1.0,
        help="With --unstable-mask-source gradient: keep a coordinate when "
             "|z0 - wall| / budget < this. 1.0 is 'the box can just reach it'; "
             "raising it admits near-misses (vit_2023: 74 at 1.0, 145 at 2.0, "
             "315 at 5.0).",
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
        batch_conversion=args.batch_conversion,
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
            safe_name = safe_name.replace("/", "_").replace("\\", "_")
            # Windows caps a path at 260 characters and this directory name is
            # the long, variable part of it. mkdir SUCCEEDS at the limit and
            # only the later write fails, so an over-long run looks fine until
            # its summaries turn out to be missing -- tinyimagenet's names are
            # exactly 120 and died on the 15-character arm "baseline_cerepl"
            # while the 8-character "baseline" squeaked through.
            #
            # So derive the budget from the actual parent path instead of
            # guessing a constant: whatever is left of 259 after the parent and
            # the longest file written inside (group_summary.json, 19 with its
            # separator; 26 leaves slack). A hash keeps two truncated names
            # from colliding. cifar100's names are 110 and its parents leave
            # exactly 110, so they are untouched and stay byte-identical.
            budget = 259 - len(str(rep_dir.resolve())) - 26
            if len(safe_name) > budget:
                digest = hashlib.sha1(safe_name.encode("utf-8")).hexdigest()[:8]
                safe_name = f"{safe_name[:max(8, budget - 9)]}_{digest}"
            group_dir = rep_dir / safe_name
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
