"""Is the unstable-neuron mask the right candidate set for a single lane?

`ACTFuzzer._init_state_manager` computes ONE mask for the whole group:

    compute_unstable_mask(model, lb, ub)          # lb/ub carry all B rows

so a neuron counts as a flip candidate if its pre-activation straddles zero
somewhere in the batch. Everything downstream is scoped to that set: HPGD only
aims inside it, PatternStateManager only scores novelty inside it, and the
coarse-to-fine schedule only measures its frontier inside it.

Two ways that can be the wrong set for a given lane, and they pull in opposite
directions:

  too WIDE  -- a neuron unstable in instance 7's box may be firmly stable in
               instance 42's. Aiming lane 42 at it asks for a flip its own box
               cannot deliver, which is indistinguishable from HPGD "missing".
  too NARROW -- interval propagation over-approximates, so a neuron the IBP
               call reports as stable can still flip in practice. Measured
               collateral says this happens a lot: the campaign logs ~26 flips
               per lane against ~4-8 inside the candidate set, i.e. most of the
               sign changes that actually occur are invisible to both the
               strategy and the admission rule.

This script measures both. Per instance it recomputes the mask from that
instance's OWN box and compares against the group mask, then runs one batch of
real mutations and counts where the sign flips actually landed.

    python -m act.pipeline.Shiyang.pipeline.unstable_mask_scope \\
        --category cifar100_2024 --max-instances 200 --instances 20 --device cpu
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
from act.pipeline.fuzzing.state_manager import compute_unstable_mask
from act.util.cli_utils import initialize_from_args

from act.pipeline.Shiyang.pipeline.common import initial_seeds_from_wrapped_model
from act.pipeline.Shiyang.pipeline.paper_cifar100_batch_ani import (
    SHIYANG_ROOT,
    build_mutation_weights,
)

DEFAULT_OUTPUT = SHIYANG_ROOT / "results" / "unstable_mask_scope"


def run_group(model_id, wrapped_model, args) -> dict:
    initial_seeds = initial_seeds_from_wrapped_model(wrapped_model)
    config = FuzzingConfig.from_yaml(
        max_iterations=1, timeout_seconds=1.0,
        seed_selection_strategy="energy",
        mutation_weights=build_mutation_weights(0.0, 0.5),
        coverage_strategy="GlobalCov", activation_threshold=0.1,
        perturb_mode="adaptive_perdim", perturb_scale=1.0,
        save_counterexamples=False, output_dir=Path(args.output),
        report_interval=10 ** 9, verbose=0,
        trace_level=0, trace_sample_rate=1, trace_storage="json", trace_output=None,
        stop_on_first_violation=False,
        admission_mode="state", scheduling_mode="energy",
        enable_bi_gce=False, bi_threaded=False, bi_attack_strategy=None,
    )
    fuzzer = ACTFuzzer(wrapped_model=wrapped_model, initial_seeds=initial_seeds, config=config)
    sm = fuzzer.state_manager
    cand = sm.candidate_indices.cpu()

    lb, ub = fuzzer.input_spec.materialize_box_seed()
    lb, ub = lb.to(fuzzer.device), ub.to(fuzzer.device)
    B = lb.shape[0]
    n_probe = min(args.instances, B)

    # -- per-instance masks --------------------------------------------------
    # One IBP pass per instance. The wrapped model's output spec is row-indexed
    # and rejects a 1-lane batch, so each instance's box is broadcast back to B
    # lanes: every lane carries the SAME box, which makes the union over lanes
    # equal to that one instance's own unstable set.
    per_inst_sizes, in_group, extra = [], [], []
    group_set = set(cand.tolist())
    for i in range(n_probe):
        lb_i = lb[i:i + 1].expand(B, *lb.shape[1:]).contiguous()
        ub_i = ub[i:i + 1].expand(B, *ub.shape[1:]).contiguous()
        mask_i, reason = compute_unstable_mask(fuzzer.model, lb_i, ub_i)
        if mask_i is None:
            print(f"   instance {i}: mask unavailable ({reason})")
            continue
        idx_i = set(mask_i.nonzero(as_tuple=True)[0].cpu().tolist())
        per_inst_sizes.append(len(idx_i))
        in_group.append(len(idx_i & group_set))
        extra.append(len(idx_i - group_set))

    # -- where do real flips land? -------------------------------------------
    seeds = fuzzer.seed_corpus.select(B)
    x = seeds.tensor.to(fuzzer.device)
    natural = _relu_sign_pattern_batched(fuzzer.model, x)
    mutated = fuzzer.mutation_engine.mutate(seeds)
    achieved = _relu_sign_pattern_batched(fuzzer.model, mutated)
    flipped = natural != achieved                       # [B, N]
    inside = flipped[:, cand.to(flipped.device)]
    n_flipped = flipped.sum(dim=1).float()
    n_inside = inside.sum(dim=1).float()

    result = {
        "model_id": str(model_id),
        "strategy_used": fuzzer.mutation_engine.last_strategy,
        "batch": int(B),
        "group_mask_size": int(cand.numel()),
        "total_neurons": int(natural.shape[1]),
        "instances_probed": len(per_inst_sizes),
        "per_instance_mask_size": {
            "mean": round(statistics.mean(per_inst_sizes), 1) if per_inst_sizes else None,
            "min": min(per_inst_sizes) if per_inst_sizes else None,
            "max": max(per_inst_sizes) if per_inst_sizes else None,
        },
        # How much of the group mask a single instance actually uses, and
        # whether an instance has unstable neurons the group mask misses.
        "share_of_group_mask": (round(statistics.mean(in_group) / cand.numel(), 4)
                                if per_inst_sizes else None),
        "unstable_for_instance_but_not_in_group_mask": {
            "mean": round(statistics.mean(extra), 1) if extra else None,
            "max": max(extra) if extra else None,
        },
        "flips_per_lane": {
            "total": round(float(n_flipped.mean()), 2),
            "inside_candidates": round(float(n_inside.mean()), 2),
            "outside_candidates": round(float((n_flipped - n_inside).mean()), 2),
            "inside_share": round(float((n_inside.sum() / n_flipped.sum().clamp(min=1))), 4),
        },
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", default="cifar100_2024")
    parser.add_argument("--max-instances", type=int, default=200)
    parser.add_argument("--instances", type=int, default=20,
                        help="How many instances to recompute a private mask for. "
                             "Each costs one IBP pass over the network.")
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
        categories=[args.category], max_instances=args.max_instances)
    results = [run_group(mid, wm, args)
               for mid, wm in synthesize_models_from_specs(spec_results).items()]

    path = out_dir / f"mask_scope_{args.category}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args) | {"output": str(out_dir)}, "groups": results}, f, indent=2)
    print(f"\n[mask_scope] written to {path}")


if __name__ == "__main__":
    main()
