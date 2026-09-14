"""Does an HPGD shift into the counterexample region give PGD a better start?

Every earlier measurement here scored HPGD on its own output, which is the
wrong thing to score: in the pipeline HPGD does not produce counterexamples,
it RELOCATES the search so that PGD attacks from somewhere else. A single HPGD
step followed by a violation check therefore measures a job HPGD is not doing,
and it duly returned 0% everywhere.

This measures the two-stage thing directly, from clean inputs, with the arms
matched on total gradient steps:

    pgd_only     N+shift steps of PGD from the original input. The control gets
                 the shift budget too, so a win for the other arms cannot be
                 "it took more steps".
    ce_then_pgd  `shift` steps of HPGD aimed at the sign pattern this instance's
                 recorded counterexamples agree on, then N steps of PGD.
    rand_then_pgd  the same shift budget aimed at a RANDOM pattern of the same
                 Hamming displacement, then N steps of PGD. The control that
                 separates "the counterexample region" from "any relocation":
                 if these two match, the counterexample information is doing
                 nothing and only the displacement matters.

Reported per arm: violation rate (the start is clean, so this is an attack
success rate outright) and the mean severity reached.

    python -m act.pipeline.Shiyang.pipeline.hpgd_then_pgd \\
        --ce-prior <ce_prior.pt> --category cifar100_2024 --max-instances 200
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

DEFAULT_OUTPUT = SHIYANG_ROOT / "results" / "hpgd_then_pgd"


class _Seeds:
    def __init__(self, tensor, original_index, original_tensor=None):
        self.tensor = tensor
        self.original_tensor = original_tensor if original_tensor is not None else tensor
        self.original_index = original_index

    def __len__(self):
        return self.tensor.shape[0]


def run_group(model_id, wrapped_model, args) -> dict:
    initial_seeds = initial_seeds_from_wrapped_model(wrapped_model)
    config = FuzzingConfig.from_yaml(
        max_iterations=1, timeout_seconds=0.1,
        seed_selection_strategy="energy",
        mutation_weights=build_mutation_weights(0.0, 0.5, absorb_non_pgd=True),
        coverage_strategy="GlobalCov", activation_threshold=0.1,
        perturb_mode="adaptive_perdim", perturb_scale=1.0,
        save_counterexamples=False, output_dir=Path(args.output),
        report_interval=10 ** 9, verbose=0,
        trace_level=0, trace_sample_rate=1, trace_storage="json", trace_output=None,
        stop_on_first_violation=False,
        admission_mode="state", scheduling_mode="energy",
        enable_bi_gce=False, bi_threaded=False, bi_attack_strategy=None,
        unstable_mask_scope="per_instance",
    )
    fuzzer = ACTFuzzer(wrapped_model=wrapped_model, initial_seeds=initial_seeds, config=config)
    sm, hpgd = fuzzer.state_manager, fuzzer._hpgd_strategy
    pgd = fuzzer.mutation_engine.strategies.get("pgd")
    cand = sm.candidate_indices.to(fuzzer.device)

    prior = {}
    if args.ce_prior:
        blob = torch.load(args.ce_prior, map_location="cpu", weights_only=False)
        prior = blob.get(str(model_id), {})
    if not prior:
        return {"model_id": str(model_id), "covered": 0}

    B = len(initial_seeds)
    lb, ub = fuzzer.input_spec.materialize_box_seed()
    lb, ub = lb.to(fuzzer.device), ub.to(fuzzer.device)
    x0 = torch.stack([s.tensor[0] for s in initial_seeds]).to(fuzzer.device)
    oi = torch.arange(B, device=fuzzer.device)
    lanes = sorted(i for i in prior.keys() if i < B)
    seeds = _Seeds(x0, oi)

    # One shared local box for every arm, so the search volume is identical.
    perturb = torch.max(x0 - lb, ub - x0)
    hpgd.perturb_size = perturb
    if pgd is not None:
        pgd.perturb_size = perturb
        pgd.num_steps = args.pgd_steps

    natural = _relu_sign_pattern_batched(fuzzer.model, x0)

    def violations(inp):
        with torch.no_grad():
            fuzzer.mutation_engine._pending_rows = oi
            out = fuzzer.model(inp)
        y = out["output"] if isinstance(out, dict) else out
        mask, _ = fuzzer.property_checker.check(inputs=inp, outputs=y,
                                                seeds=_Seeds(inp, oi, x0))
        sev = fuzzer.output_spec.severity(y)
        return mask, sev.detach()

    def shift(target_full, mask):
        hpgd.num_steps = args.shift_steps
        hpgd.target_override = target_full
        hpgd.target_override_mask = mask
        try:
            out = hpgd.mutate(x0, fuzzer.model)
        finally:
            hpgd.target_override = None
            hpgd.target_override_mask = None
        return fuzzer.mutation_engine._project(out, seeds)

    def attack(start, steps):
        if pgd is None:
            return start
        pgd.num_steps = steps
        out = pgd.mutate(start, fuzzer.mutation_engine.attack_model,
                         fuzzer.mutation_engine.activation_map, rows=oi)
        return fuzzer.mutation_engine._project(out, seeds)

    lane_mask = torch.zeros(B, dtype=torch.bool, device=fuzzer.device)
    lane_mask[torch.tensor(lanes, device=fuzzer.device)] = True

    # CE target: the sign the counterexamples agree on, on this lane's own
    # candidate axis; everything else stays at the seed's natural sign.
    ce_target = natural.clone()
    for b in lanes:
        ce_target[b, cand] = prior[b]["ce_sign"].to(fuzzer.device)[cand].to(ce_target.dtype)
    d_ce = (ce_target[:, cand] != natural[:, cand]).sum(dim=1)

    rand_target = natural.clone()
    for b in lanes:
        k = int(d_ce[b])
        if k > 0:
            pick = torch.randperm(cand.numel(), device=fuzzer.device)[:k]
            rand_target[b, cand[pick]] *= -1

    arms = {}
    start_mask, _ = violations(x0)
    arms["start"] = start_mask
    arms["pgd_only"] = violations(attack(x0, args.pgd_steps + args.shift_steps))
    arms["ce_then_pgd"] = violations(attack(shift(ce_target, lane_mask), args.pgd_steps))
    arms["rand_then_pgd"] = violations(attack(shift(rand_target, lane_mask), args.pgd_steps))

    idx = torch.tensor(lanes, device=fuzzer.device)
    res = {"model_id": str(model_id), "covered": len(lanes),
           "d_ce_mean": round(float(d_ce[idx].float().mean()), 1),
           "pgd_steps": args.pgd_steps, "shift_steps": args.shift_steps,
           "start_violation_rate": round(float(start_mask[idx].float().mean()), 4),
           "arms": {}}
    for k in ("pgd_only", "ce_then_pgd", "rand_then_pgd"):
        mask, sev = arms[k]
        res["arms"][k] = {"violation_rate": round(float(mask[idx].float().mean()), 4),
                          "mean_severity": round(float(sev[idx].float().mean()), 6)}
    print(json.dumps(res, indent=2))
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--category", default="cifar100_2024")
    p.add_argument("--max-instances", type=int, default=200)
    p.add_argument("--ce-prior", required=True)
    p.add_argument("--pgd-steps", type=int, default=50)
    p.add_argument("--shift-steps", type=int, default=10)
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--device", choices=["cpu", "cuda", "gpu"], default="cpu")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = p.parse_args()

    out = Path(args.output)
    if not out.is_absolute():
        out = SHIYANG_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    initialize_from_args(args)

    spec_results = VNNLibSpecCreator().create_specs_for_data_model_pairs(
        categories=[args.category], max_instances=args.max_instances)
    results = [run_group(mid, wm, args)
               for mid, wm in synthesize_models_from_specs(spec_results).items()]
    path = out / f"hpgd_then_pgd_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "groups": results}, f, indent=2)
    print(f"\n[hpgd_then_pgd] written to {path}")


if __name__ == "__main__":
    main()
