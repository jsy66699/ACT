"""Does steering HPGD at counterexample-discriminative neurons beat flipping at random?

Every neuron-selection criterion tried so far ranks by how much a neuron MOVES
(balance under random points of the box) or by nothing at all (uniform). None
of them changed distinct-instances. The criterion that repair-side fault
localisation actually uses -- contrast the activations of FAILING against
PASSING inputs -- was unavailable, because an instance that has no
counterexample yet cannot supply the failing side, and another instance's
counterexamples do not transfer (different box, different unstable set;
measured, cross-instance targets land like random ones).

This script asks the question where the data DOES exist: on instances the
fuzzer has already cracked. It is deliberately a conditional question --
"given a counterexample, does CE-informed flip selection raise the violation
rate" -- not "can it crack new instances". If the answer is no even here, then
pattern-space steering has no headroom to recover, because this is the most
favourable case it will ever get.

Three arms, run on the SAME seeds with the SAME projection (HPGDMutation via
`target_override`), so they are paired and matched on displacement k:

    random      k random candidates flipped -- what production does.
    ce_toward   the k most CE-discriminative neurons, pushed to the sign the
                counterexamples have.
    ce_against  the same k neurons, pushed to the opposite sign.

`ce_against` is the internal control that makes the result interpretable: if
aiming at the CE side helps, aiming away must hurt. Three equal arms would mean
the selected neurons carry no directional information and any difference
between random and ce_toward is displacement, not guidance.

Scoring (per instance, over its own candidate axis):

    score_j = | P(sign_j = +1 | CE samples) - P(sign_j = +1 | non-CE samples) |

the continuous form of a spectrum-based suspiciousness. Both populations come
from PatternStateManager's registry, which already records `is_ce` per entry.

    python -m act.pipeline.Shiyang.pipeline.hpgd_ce_guided_flips \\
        --category cifar100_2024 --max-instances 200 --warmup 60 --batches 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

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

DEFAULT_OUTPUT = SHIYANG_ROOT / "results" / "hpgd_ce_guided"


def ce_discriminative_scores(sm, candidates):
    """{instance: (score[C], ce_sign[C])} for every instance with both a
    counterexample and a non-counterexample recorded."""
    by_inst = {}
    for node in sm._registry:
        inst = int(node.payload["original_index"].reshape(-1)[0])
        bucket = by_inst.setdefault(inst, {"ce": [], "non": []})
        bucket["ce" if node.payload.get("is_ce") else "non"].append(node.point)

    out = {}
    for inst, bucket in by_inst.items():
        if not bucket["ce"] or not bucket["non"]:
            continue
        ce = torch.stack(bucket["ce"]).float()
        non = torch.stack(bucket["non"]).float()
        p_ce = (ce > 0).float().mean(dim=0)
        p_non = (non > 0).float().mean(dim=0)
        score = (p_ce - p_non).abs()
        # Sign the counterexamples agree on, where they agree at all.
        ce_sign = torch.where(p_ce >= 0.5, torch.ones_like(p_ce), -torch.ones_like(p_ce))
        out[inst] = (score, ce_sign, len(bucket["ce"]), len(bucket["non"]))
    return out


def run_group(model_id, wrapped_model, args) -> dict:
    initial_seeds = initial_seeds_from_wrapped_model(wrapped_model)
    config = FuzzingConfig.from_yaml(
        max_iterations=args.max_iterations, timeout_seconds=args.warmup,
        seed_selection_strategy="energy",
        mutation_weights=build_mutation_weights(0.0, args.hpgd_weight, absorb_non_pgd=True),
        coverage_strategy="GlobalCov", activation_threshold=0.1,
        perturb_mode="adaptive_perdim", perturb_scale=1.0,
        save_counterexamples=False, output_dir=Path(args.output),
        report_interval=10 ** 9, verbose=args.verbose,
        trace_level=0, trace_sample_rate=1, trace_storage="json", trace_output=None,
        stop_on_first_violation=False,
        admission_mode="state", scheduling_mode="energy",
        enable_bi_gce=False, bi_threaded=False, bi_attack_strategy=None,
        ce_parent_replacement=True,
        unstable_mask_scope=args.unstable_mask,
    )
    fuzzer = ACTFuzzer(wrapped_model=wrapped_model, initial_seeds=initial_seeds, config=config)
    print(f"\n[ce_guided] {model_id!r}: {args.warmup}s warmup to collect counterexamples")
    fuzzer.fuzz()

    sm, hpgd = fuzzer.state_manager, fuzzer._hpgd_strategy
    candidates = (hpgd.candidate_indices if hpgd.candidate_indices is not None
                  else sm.candidate_indices).to(fuzzer.device)
    scored = ce_discriminative_scores(sm, candidates)
    if not scored:
        print("[ce_guided] no instance has both a CE and a non-CE recorded; "
              "nothing to score. Raise --warmup.")
        return {"model_id": str(model_id), "scored_instances": 0}
    print(f"[ce_guided] {len(scored)} instance(s) with counterexamples; "
          f"k={args.flips}, candidate axis {candidates.numel()}")

    batch = len(initial_seeds)
    arms = {"random": [], "ce_toward": [], "ce_against": []}

    def violations_of(inputs, oi, seeds):
        with torch.no_grad():
            fuzzer.mutation_engine._pending_rows = oi
            out = fuzzer.model(inputs)
        y = out["output"] if isinstance(out, dict) else out
        mask, _ = fuzzer.property_checker.check(
            inputs=inputs, outputs=y, seeds=_FakeSeeds(inputs, oi))
        return mask

    def measure(x, target_full, oi, lanes, seeds, start_violated):
        natural = _relu_sign_pattern_batched(fuzzer.model, x)
        hpgd.target_override = target_full
        hpgd.target_override_mask = torch.zeros(x.shape[0], dtype=torch.bool,
                                                device=x.device).index_fill_(0, lanes, True)
        try:
            mutated = hpgd.mutate(x, fuzzer.model)
        finally:
            hpgd.target_override = None
            hpgd.target_override_mask = None
        # HPGD clamps to its own local box; the InputSpec box is a separate,
        # tighter constraint that MutationEngine applies after every mutate.
        # Skipping it lets samples drift outside the instance's feasible region,
        # where PropertyChecker rejects the violation as infeasible.
        mutated = fuzzer.mutation_engine._project(mutated, seeds)
        # PropertyChecker needs the per-lane input-feasibility metadata that
        # MutationEngine's forward hooks attach, and those only fire when
        # _pending_rows is set -- normally by mutate(). This measurement calls
        # HPGD directly, so it has to arm them itself, or the checker refuses
        # the batch ("requires lane-aware input feasibility metadata").
        with torch.no_grad():
            fuzzer.mutation_engine._pending_rows = oi
            out = fuzzer.model(mutated)
        y = out["output"] if isinstance(out, dict) else out
        violated, _ = fuzzer.property_checker.check(
            inputs=mutated, outputs=y, seeds=_FakeSeeds(mutated, oi))
        achieved = _relu_sign_pattern_batched(fuzzer.model, mutated)
        nat_c, tgt_c, ach_c = (natural[:, candidates], target_full[:, candidates],
                               achieved[:, candidates])
        asked = tgt_c != nat_c
        landed = asked & (ach_c == tgt_c)
        rows = []
        for b in lanes.tolist():
            d = int(asked[b].sum())
            if d == 0:
                continue
            rows.append({
                "d": d,
                "hit_rate": int(landed[b].sum()) / d,
                "violated": float(violated[b]),
                "start_violated": float(start_violated[b]),
                "moved": int((ach_c[b] != nat_c[b]).sum()),
            })
        return rows

    for _ in range(args.batches):
        seeds = fuzzer.seed_corpus.select(batch)
        oi = seeds.original_index.to(fuzzer.device)
        # The corpus of a cracked instance is dominated by counterexample
        # descendants (energy 100 against an ordinary seed's 10), so a seed
        # drawn from it usually violates BEFORE anything is mutated. Measuring
        # violation after the flips without this control credits the flips for
        # a property the starting point already had.
        x = (seeds.original_tensor if args.start == "original" else seeds.tensor)
        x = x.to(fuzzer.device)
        start_viol = violations_of(x, oi, seeds)
        natural = _relu_sign_pattern_batched(fuzzer.model, x)
        nat_c = natural[:, candidates]

        lanes = [b for b in range(batch) if int(oi[b]) in scored]
        if not lanes:
            continue
        lanes_t = torch.tensor(lanes, device=x.device)

        t_rand = natural.clone()
        t_toward = natural.clone()
        t_against = natural.clone()
        for b in lanes:
            score, ce_sign, _, _ = scored[int(oi[b])]
            score = score.to(x.device)
            ce_sign = ce_sign.to(x.device)
            # Rank only the positions where the wanted sign actually DIFFERS
            # from this seed's own, so all three arms ask for exactly k real
            # flips. Ranking the raw top-k instead lets ce_* quietly request
            # fewer flips (the CE sign often already matches), which would make
            # its higher landing rate a displacement artefact.
            def _topk_where(diff_mask):
                masked = torch.where(diff_mask, score, torch.full_like(score, -1.0))
                n = int(diff_mask.sum())
                if n == 0:
                    return None
                return torch.topk(masked, min(args.flips, n)).indices

            top_t = _topk_where(ce_sign != nat_c[b])
            top_a = _topk_where(-ce_sign != nat_c[b])
            pick = torch.randperm(candidates.numel(), device=x.device)[:args.flips]
            t_rand[b, candidates[pick]] *= -1
            if top_t is not None:
                t_toward[b, candidates[top_t]] = ce_sign[top_t].to(t_toward.dtype)
            if top_a is not None:
                t_against[b, candidates[top_a]] = (-ce_sign[top_a]).to(t_against.dtype)

        arms["random"] += measure(x, t_rand, oi, lanes_t, seeds, start_viol)
        arms["ce_toward"] += measure(x, t_toward, oi, lanes_t, seeds, start_viol)
        arms["ce_against"] += measure(x, t_against, oi, lanes_t, seeds, start_viol)

    def summarize(rows):
        if not rows:
            return {"n": 0}
        mean = lambda k, rs=None: statistics.mean(r[k] for r in (rs or rows))
        fresh = [r for r in rows if r["start_violated"] == 0.0]
        already = [r for r in rows if r["start_violated"] == 1.0]
        return {
            "n": len(rows), "d": round(mean("d"), 1),
            "hit_rate": round(mean("hit_rate"), 4),
            "moved": round(mean("moved"), 1),
            "start_violation_rate": round(mean("start_violated"), 4),
            "violation_rate": round(mean("violated"), 4),
            # The number that actually says "this attack worked": lanes whose
            # starting point did NOT violate, that violate after the flips.
            "new_violation_rate": (round(mean("violated", fresh), 4) if fresh else None),
            "n_fresh": len(fresh),
            # Lanes that started violating and still do -- says the mutation
            # stayed inside the region, not that it found anything.
            "retained_rate": (round(mean("violated", already), 4) if already else None),
            "n_already": len(already),
        }

    result = {"model_id": str(model_id), "scored_instances": len(scored),
              "flips": args.flips, "batch": batch,
              "arms": {k: summarize(v) for k, v in arms.items()}}
    print(json.dumps(result, indent=2))
    return result


class _FakeSeeds:
    """What PropertyChecker.check() reads off a seed batch: original_index for
    the spec row, and original_tensor when it materialises a Counterexample.
    Passing the mutated input as the original is fine here -- this script never
    keeps the counterexample objects, only the boolean mask."""

    def __init__(self, tensor, original_index):
        self.tensor = tensor
        self.original_tensor = tensor
        self.original_index = original_index

    def __len__(self):
        return self.tensor.shape[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--category", default="cifar100_2024")
    p.add_argument("--max-instances", type=int, default=200)
    p.add_argument("--warmup", type=float, default=60.0)
    p.add_argument("--batches", type=int, default=10)
    p.add_argument("--flips", type=int, default=10,
                   help="k, identical in all three arms so they are matched on displacement.")
    p.add_argument("--start", choices=["corpus", "original"], default="corpus",
                   help="Where the measured flips start. 'corpus' uses the seeds "
                        "scheduling would actually hand HPGD -- for a cracked "
                        "instance those are mostly counterexample descendants, so "
                        "report new_violation_rate, not violation_rate. 'original' "
                        "starts from the clean input, which makes the violation "
                        "rate an attack success rate outright.")
    p.add_argument("--hpgd-weight", type=float, default=0.5)
    p.add_argument("--unstable-mask", choices=["row0", "union", "per_instance"],
                   default="per_instance")
    p.add_argument("--max-iterations", type=int, default=10_000_000)
    p.add_argument("--verbose", type=int, default=1)
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--device", choices=["cpu", "cuda", "gpu"], default="cpu")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = p.parse_args()

    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = SHIYANG_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    initialize_from_args(args)

    spec_results = VNNLibSpecCreator().create_specs_for_data_model_pairs(
        categories=[args.category], max_instances=args.max_instances)
    results = [run_group(mid, wm, args)
               for mid, wm in synthesize_models_from_specs(spec_results).items()]

    path = out_dir / f"ce_guided_{args.category}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args) | {"output": str(out_dir)}, "groups": results}, f, indent=2)
    print(f"\n[ce_guided] written to {path}")


if __name__ == "__main__":
    main()
