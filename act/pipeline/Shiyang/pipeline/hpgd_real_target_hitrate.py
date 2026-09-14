"""Where should HPGD aim -- and is landing even the goal?

Background. HPGD builds its target by flipping k bits of the seed's own ReLU
sign pattern INDEPENDENTLY. Nothing checks the result is jointly satisfiable,
so a low landing rate is unattributable: "the projection is weak" and "there
was nothing there to reach" produce the same number (the
`feedback_real_targets_only` trap). The first version of this script settled
that -- at a MATCHED displacement, a target some sample actually reached is
hit ~100% and exactly, while a fabricated one at the same distance is hit
~6-12% and never exactly. The optimizer is fine; the targets are not.

But landing is NOT the objective, and optimising it would be a mistake:

  * A pattern in PatternStateManager's registry is by definition one that was
    ALREADY ADMITTED. Steering to it precisely produces an achieved pattern
    admission rejects -- 100% hit rate, 0% novelty. Today's HPGD earns its
    keep by MISSING: 78% of its landings are novel states.
  * A registry pattern from ANOTHER instance is not "real" for this seed at
    all. Same weights and same eps still means a different input box, hence a
    different set of realizable sign patterns; a cross-instance target is as
    fabricated as a random flip, only farther away.

So the design question is not "how do we hit more" but "how do we name a
target that is plausibly FEASIBLE and NOT yet visited". This script measures
candidate answers on both axes at once.

Arms (all run the SAME projection -- HPGDMutation.mutate() with
`target_override` -- on the SAME seeds, so they are paired):

    real_same_near   nearest registry pattern from the SAME instance. Feasible
                     (that lane's own box realized it) but already visited:
                     the novelty floor. Typically ONE bit away -- the registry
                     covers the local neighbourhood that densely.
    real_same_far    the farthest same-instance one (--far-quantile), which is
                     what the interpolation arms aim between.
    interp_RHO       start from that same neighbour P and flip only a random
                     RHO-share of the positions where P differs from the seed.
                     Those positions are jointly realizable transitions (both
                     endpoints are real), so a partial move names a cell
                     BETWEEN two real states -- which has a structural reason
                     to be non-empty that k random flips do not -- and which
                     is not itself in the registry.
    real_cross       nearest registry pattern from a DIFFERENT instance, to
                     put a number on the objection above instead of assuming it.
    synth_matched_d  d random flips, d taken from real_same. Feasibility
                     control at matched displacement (`feedback_matched_comparisons`).
    synth_kK         what production does today.

Reported per arm: hit rate over the asked flips, exact arrival, residual/d
(>1 means the projection ended FARTHER from its own target than it started),
collateral flips, and -- the one that matters -- `novel`, the share of
achieved patterns admission would accept, queried through
PatternStateManager.seen_mask() without recording anything.

Usage:

    python -m act.pipeline.Shiyang.pipeline.hpgd_real_target_hitrate \\
        --category cifar100_2024 --max-instances 200 --warmup 60 \\
        --batches 10 --device cpu
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import torch

from act.front_end.model_synthesis import synthesize_models_from_specs
from act.front_end.vnnlib_loader.create_specs import VNNLibSpecCreator
from act.pipeline.fuzzing.actfuzzer import ACTFuzzer, FuzzingConfig
from act.pipeline.fuzzing.mutations import _relu_sign_pattern_batched
from act.util.cli_utils import initialize_from_args

from act.pipeline.Shiyang.pipeline.common import initial_seeds_from_wrapped_model
from act.pipeline.Shiyang.pipeline.paper_cifar100_batch_ani import (
    SHIYANG_ROOT,
    build_mutation_weights,
)


DEFAULT_OUTPUT = SHIYANG_ROOT / "results" / "hpgd_real_target_hitrate"


def _measure(fuzzer, sm, hpgd, x, target_full, candidates, original_index):
    """Run one projection at `target_full` and score it per sample."""
    natural = _relu_sign_pattern_batched(fuzzer.model, x, binning=fuzzer.binning)
    hpgd.target_override = target_full
    try:
        mutated = hpgd.mutate(x, fuzzer.model)
    finally:
        hpgd.target_override = None
    achieved = _relu_sign_pattern_batched(fuzzer.model, mutated,
                                          binning=fuzzer.binning)

    # Pure queries -- seen_mask records nothing, so every arm is scored against
    # the same frozen post-warmup state and the arms stay comparable.
    novel = ~sm.seen_mask(achieved, original_index)
    target_novel = ~sm.seen_mask(target_full, original_index)

    nat_c = natural[:, candidates]
    tgt_c = target_full[:, candidates]
    ach_c = achieved[:, candidates]
    asked = tgt_c != nat_c
    landed = asked & (ach_c == tgt_c)
    collateral = (ach_c != nat_c) & ~asked

    rows = []
    for b in range(x.shape[0]):
        d = int(asked[b].sum())
        if d == 0:
            continue  # target is the seed's own pattern; nothing was asked
        hit = int(landed[b].sum())
        rows.append({
            "d": d,
            "hit_rate": hit / d,
            "exact": float(hit == d),
            "residual": int((ach_c[b] != tgt_c[b]).sum()),
            "collateral": int(collateral[b].sum()),
            "novel": float(novel[b]),
            "target_novel": float(target_novel[b]),
        })
    return rows


def _summarize(rows):
    if not rows:
        return {"n": 0}
    mean = lambda k: statistics.mean(r[k] for r in rows)
    return {
        "n": len(rows),
        "d": round(mean("d"), 1),
        "hit_rate": round(mean("hit_rate"), 4),
        "exact": round(mean("exact"), 4),
        "residual_over_d": round(mean("residual") / max(mean("d"), 1e-9), 3),
        "collateral": round(mean("collateral"), 1),
        "novel": round(mean("novel"), 4),
        "target_novel": round(mean("target_novel"), 4),
    }


def _bucket(rows):
    buckets = defaultdict(list)
    for r in rows:
        d = r["d"]
        buckets["1-9" if d < 10 else "10-99" if d < 100 else
                "100-999" if d < 1000 else "1000+"].append(r)
    return {k: _summarize(v) for k, v in sorted(buckets.items())}


def _pick(points, node_inst, nat_row, inst, same: bool, q: float = 0.0) -> Optional[int]:
    """Index of a registry pattern at distance quantile `q` from `nat_row`,
    among nodes of the same instance (same=True) or of any other (same=False).
    q=0 is the nearest, q=1 the farthest. Distance 0 is skipped: that is the
    seed's own current state, which asks for nothing. None when the pool is
    empty.

    The quantile exists because "nearest" turned out to be degenerate: the
    registry covers a seed's immediate neighbourhood so densely that the
    closest same-instance state is typically ONE bit away, leaving nothing to
    interpolate. Testing "a cell between two real states" needs two real
    states that are actually apart."""
    pool = (node_inst == inst) if same else (node_inst != inst)
    idx = pool.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return None
    d = (points[idx] != nat_row).sum(dim=1)
    keep = (d > 0).nonzero(as_tuple=True)[0]
    if keep.numel() == 0:
        return None
    idx, d = idx[keep], d[keep]
    order = torch.argsort(d)
    rank = min(int(round(q * (order.numel() - 1))), order.numel() - 1)
    return int(idx[order[rank]])


def run_group(model_id, wrapped_model, args) -> dict[str, Any]:
    initial_seeds = initial_seeds_from_wrapped_model(wrapped_model)
    if not initial_seeds:
        raise RuntimeError(f"No initial seeds for {model_id!r}")

    config = FuzzingConfig.from_yaml(
        max_iterations=args.max_iterations,
        timeout_seconds=args.warmup,
        seed_selection_strategy="energy",
        mutation_weights=build_mutation_weights(0.0, args.hpgd_weight),
        coverage_strategy="GlobalCov",
        activation_threshold=0.1,
        perturb_mode="adaptive_perdim",
        perturb_scale=args.perturb_scale,
        save_counterexamples=False,
        output_dir=Path(args.output),
        report_interval=args.report_interval,
        verbose=args.verbose,
        trace_level=0,
        trace_sample_rate=1,
        trace_storage="json",
        trace_output=None,
        stop_on_first_violation=False,
        admission_mode="state",       # what populates the registry
        scheduling_mode="energy",
        enable_bi_gce=False,
        bi_threaded=False,
        bi_attack_strategy=None,
        hpgd_flip_frac=args.hpgd_flip_frac,
        state_bins=args.state_bins,
        state_bin_tau=args.state_bin_tau,
        hpgd_margin=args.hpgd_margin,
    )

    fuzzer = ACTFuzzer(wrapped_model=wrapped_model, initial_seeds=initial_seeds, config=config)
    print(f"\n[hpgd_hitrate] {model_id!r}: {args.warmup}s warmup to fill the registry")
    fuzzer.fuzz()

    sm, hpgd = fuzzer.state_manager, fuzzer._hpgd_strategy
    if args.probe_margin is not None and hpgd is not None:
        # Changed AFTER the warmup on purpose. `--hpgd-margin` also steers the
        # fuzzing that fills the registry, so comparing two margins that way
        # compares two different registries: measured, margin 0 halved the
        # distance to the farthest real state (d 63.3 -> 29.9), and every arm
        # whose d is not fixed by construction then reads as "closer target",
        # not as "better projection". This knob leaves the registry identical
        # and varies only the projection under test.
        print(f"[hpgd_hitrate] probe margin {hpgd.margin} -> {args.probe_margin} "
              f"(registry built at {hpgd.margin})")
        hpgd.margin = float(args.probe_margin)
    if sm is None or hpgd is None:
        raise RuntimeError("state manager / hpgd strategy missing -- check --hpgd-weight")
    if len(sm) == 0:
        raise RuntimeError("registry empty after warmup: no real target exists. Raise --warmup.")

    candidates = (hpgd.candidate_indices if hpgd.candidate_indices is not None
                  else sm.candidate_indices).to(fuzzer.device)
    default_k = (max(1, round(args.hpgd_flip_frac * candidates.numel()))
                 if args.hpgd_flip_frac else hpgd.flip_count)

    # The registry as two tensors: the patterns, and which instance each came
    # from. observe_batch appends to _registry WITHOUT inserting into the
    # BK-tree (the tree is only built on the diversity_threshold>1 path), so a
    # linear scan is not a shortcut here -- it is the only index there is.
    registry = sm._registry
    points = torch.stack([n.point for n in registry]).to(fuzzer.device)
    node_inst = torch.tensor([int(n.payload["original_index"].item()) for n in registry],
                             device=fuzzer.device)
    print(f"[hpgd_hitrate] registry={len(registry)} patterns over "
          f"{int(node_inst.unique().numel())} instances; "
          f"{candidates.numel()} candidate neurons")

    # Every probe runs the WRAPPED model, whose row-indexed output spec accepts
    # exactly B lanes ("y_true carries 100 spec rows but the batch has 50").
    batch = len(initial_seeds)
    if args.batch_size is not None and args.batch_size != batch:
        raise ValueError(f"--batch-size {args.batch_size} != this group's B={batch}")

    arms: dict[str, list] = defaultdict(list)
    rhos = [float(r) for r in args.interp.split(",") if r.strip()]
    for _ in range(args.batches):
        seeds = fuzzer.seed_corpus.select(batch)
        x = seeds.tensor.to(fuzzer.device)
        oi = seeds.original_index.to(fuzzer.device)
        natural = _relu_sign_pattern_batched(fuzzer.model, x, binning=fuzzer.binning)
        nat_c = natural[:, candidates]

        near_same, far_same, near_cross = {}, {}, {}
        for b in range(x.shape[0]):
            inst = int(oi[b].item())
            j = _pick(points, node_inst, nat_c[b], inst, same=True, q=0.0)
            if j is not None:
                near_same[b] = j
            j = _pick(points, node_inst, nat_c[b], inst, same=True, q=args.far_quantile)
            if j is not None:
                far_same[b] = j
            j = _pick(points, node_inst, nat_c[b], inst, same=False, q=0.0)
            if j is not None:
                near_cross[b] = j

        def target_from(neigh: dict, rho: Optional[float]):
            """rho=None: the neighbour itself. Otherwise flip a random rho-share
            of the positions where it differs from this seed."""
            t = natural.clone()
            for b, j in neigh.items():
                p = points[j]
                if rho is None:
                    t[b, candidates] = p.to(t.dtype)
                    continue
                diff = (p != nat_c[b]).nonzero(as_tuple=True)[0]
                if diff.numel() == 0:
                    continue
                take = max(1, int(round(rho * diff.numel())))
                pick = diff[torch.randperm(diff.numel(), device=x.device)[:take]]
                t[b, candidates[pick]] *= -1
            return t

        same_target = target_from(near_same, None)
        arms["real_same_near"] += _measure(fuzzer, sm, hpgd, x, same_target, candidates, oi)
        arms["real_same_far"] += _measure(
            fuzzer, sm, hpgd, x, target_from(far_same, None), candidates, oi)
        # Interpolation is measured against the FAR neighbour: between two real
        # states that are genuinely apart, there is something in between.
        for rho in rhos:
            arms[f"interp_{rho:g}"] += _measure(
                fuzzer, sm, hpgd, x, target_from(far_same, rho), candidates, oi)
        arms["real_cross"] += _measure(
            fuzzer, sm, hpgd, x, target_from(near_cross, None), candidates, oi)

        # synthetic at real_same's displacement, per sample
        d_same = (same_target[:, candidates] != nat_c).sum(dim=1)
        synth = natural.clone()
        for b in range(x.shape[0]):
            d = int(d_same[b])
            if d == 0:
                continue
            pick = torch.randperm(candidates.numel(), device=x.device)[:d]
            synth[b, candidates[pick]] *= -1
        arms["synth_matched_d"] += _measure(fuzzer, sm, hpgd, x, synth, candidates, oi)

        fixed = natural.clone()
        for b in range(x.shape[0]):
            pick = torch.randperm(candidates.numel(), device=x.device)[:default_k]
            fixed[b, candidates[pick]] *= -1
        arms[f"synth_k{int(default_k)}"] += _measure(
            fuzzer, sm, hpgd, x, fixed, candidates, oi)

    result = {
        "model_id": str(model_id),
        "batch_size": batch,
        "batches": args.batches,
        "warmup_s": args.warmup,
        "registry_size": len(registry),
        "num_candidates": int(candidates.numel()),
        "default_k": int(default_k),
        "arms": {k: _summarize(v) for k, v in arms.items()},
        "by_distance": {k: _bucket(v) for k, v in arms.items()},
    }
    print(json.dumps(result["arms"], indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", default="cifar100_2024")
    parser.add_argument("--max-instances", type=int, default=200)
    parser.add_argument("--warmup", type=float, default=60.0)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Normally omitted -- the measurement batch must be the "
                             "group's B, because the wrapped model's output spec is "
                             "row-indexed.")
    parser.add_argument("--far-quantile", type=float, default=1.0,
                        help="Which same-instance registry pattern the interpolation "
                             "arms aim between: 0 = nearest, 1 = farthest (default). "
                             "The nearest is typically 1 bit away, which leaves nothing "
                             "to interpolate.")
    parser.add_argument("--interp", default="0.25,0.5,0.75",
                        help="Comma-separated shares of the difference set to flip when "
                             "aiming BETWEEN the seed and its nearest real same-instance "
                             "neighbour.")
    parser.add_argument("--hpgd-weight", type=float, default=0.5)
    parser.add_argument("--hpgd-flip-frac", type=float, default=None)
    parser.add_argument("--state-bins", type=int, choices=[2, 3], default=2,
                        help="State partition the registry, the targets and the "
                             "hinge all use. 3 = the smooth-activation split at "
                             "z = -tau, +tau (two +-1 coordinates per neuron).")
    parser.add_argument("--state-bin-tau", type=float, default=1.0)
    parser.add_argument("--probe-margin", type=float, default=None,
                        help="Override the hinge margin for the MEASUREMENT "
                             "projections only, after the warmup has filled the "
                             "registry. This is the matched way to test a margin: "
                             "same registry, same targets, same displacements.")
    parser.add_argument("--hpgd-margin", type=float, default=0.01,
                        help="Hinge margin. 0 stops the projection the moment a "
                             "coordinate crosses its wall instead of clearing it "
                             "by `margin`, so each coordinate is cheaper to "
                             "satisfy -- but it also parks the pre-activation ON "
                             "the wall, where the code is one step-size away from "
                             "flipping back.")
    parser.add_argument("--perturb-scale", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int, default=10_000_000)
    parser.add_argument("--report-interval", type=int, default=5000)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", choices=["cpu", "cuda", "gpu"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = parser.parse_args()

    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = SHIYANG_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    initialize_from_args(args)

    spec_results = VNNLibSpecCreator().create_specs_for_data_model_pairs(
        categories=[args.category], max_instances=args.max_instances,
    )
    if not spec_results:
        raise RuntimeError(f"No VNNLIB specs for category={args.category!r}")

    results = [run_group(mid, wm, args)
               for mid, wm in synthesize_models_from_specs(spec_results).items()]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"hitrate_{args.category}_{stamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args) | {"output": str(out_dir)}, "groups": results}, f, indent=2)
    print(f"\n[hpgd_hitrate] written to {path}")


if __name__ == "__main__":
    main()
