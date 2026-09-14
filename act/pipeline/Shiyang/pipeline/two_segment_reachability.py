"""Do two-segment transitions exist among REAL observed states?

`bin_transfer_probe` asked whether a NAMED move (say high -> low) can be
reached, and got 8% on sigmoid and 0% on tanh.  That number is unattributable:
the targets were fabricated by `write_bin`, so "the projection is too weak" and
"the box never realizes that segment for that neuron" produce the same 0
(`feedback_real_targets_only`).  Enlarging the clamp box to 4x lifted it to
38%, which is evidence for the second reading, not the first.

So before engineering anything to "cross two segments", this asks whether the
crossing is a thing that happens at all.  It fills the registry with real
observed states exactly as the fuzzer does, then, per instance, looks at the
segment each neuron takes across that instance's own observed states:

    span 1   the neuron sat in one segment for every state ever observed
    span 2   it was seen in two segments (a one-wall move really happens)
    span 3   it was seen in all three -- and in particular in BOTH low and
             high, which is the two-wall move `bin_transfer_probe` could not
             manufacture

Only neurons whose BOTH coordinates are candidates can span three, since a
pinned coordinate is one the box cannot move by definition; those are counted
separately rather than mixed in.

    python -m act.pipeline.Shiyang.pipeline.two_segment_reachability \\
        --max-instances 99 --warmup 60
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from act.pipeline.fuzzing.actfuzzer import ACTFuzzer, FuzzingConfig
from act.pipeline.Shiyang.pipeline.common import initial_seeds_from_wrapped_model
from act.pipeline.Shiyang.pipeline.paper_cifar100_batch_ani import build_mutation_weights
from act.front_end.model_synthesis import synthesize_models_from_specs
from act.front_end.vnnlib_loader.create_specs import VNNLibSpecCreator


def analyse(fuzzer, args) -> None:
    sm = fuzzer.state_manager
    binning = fuzzer.binning
    if sm is None or binning is None or binning.bins != 3:
        raise RuntimeError("needs --state-bins 3 and state admission")
    if len(sm) == 0:
        raise RuntimeError("registry empty after warmup -- raise --warmup")

    cand = sm.candidate_indices.cpu()                  # coord indices into 2N
    # A neuron is fully observable only if BOTH of its coordinates are
    # candidates; with one pinned it can still move, but never across two walls.
    pos_of = {int(c): i for i, c in enumerate(cand.tolist())}
    neurons_both, neurons_one = [], []
    for j in range(binning.num_neurons):
        lo, hi = 2 * j in pos_of, 2 * j + 1 in pos_of
        (neurons_both if lo and hi else neurons_one if (lo or hi) else []).append(j)

    pts = torch.stack([n.point for n in sm._registry]).cpu()
    inst = torch.tensor([int(n.payload["original_index"].reshape(-1)[0])
                         for n in sm._registry])
    print(f"[two_seg] registry={len(pts)} states over {int(inst.unique().numel())} "
          f"instances | candidates={cand.numel()} coords | "
          f"neurons with both coords candidate={len(neurons_both)}, "
          f"one={len(neurons_one)}")

    span_hist = defaultdict(int)
    both_lowhigh = 0
    per_inst_lowhigh = defaultdict(int)
    n_inst = 0
    for i in inst.unique().tolist():
        rows = pts[inst == i]
        if rows.shape[0] < 2:
            continue
        n_inst += 1
        for j in neurons_both:
            lo = rows[:, pos_of[2 * j]] > 0
            hi = rows[:, pos_of[2 * j + 1]] > 0
            seg = lo.long() + hi.long()          # 0 low, 1 mid, 2 high
            seen = set(seg.tolist())
            span_hist[len(seen)] += 1
            if 0 in seen and 2 in seen:
                both_lowhigh += 1
                per_inst_lowhigh[i] += 1

    total = sum(span_hist.values())
    print(f"[two_seg] {n_inst} instances with >=2 observed states; "
          f"{total} (instance, neuron) pairs examined")
    for k in sorted(span_hist):
        print(f"    seen in {k} segment(s): {span_hist[k]:>7} "
              f"({span_hist[k] / max(total, 1):.2%})")
    print(f"    seen in BOTH low and high: {both_lowhigh} "
          f"({both_lowhigh / max(total, 1):.3%}) "
          f"over {len(per_inst_lowhigh)} instances")
    if per_inst_lowhigh:
        top = sorted(per_inst_lowhigh.items(), key=lambda kv: -kv[1])[:5]
        print("    per-instance top: " + ", ".join(f"inst{i}:{c}" for i, c in top))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="eran_sigmoid_tanh_mlp")
    p.add_argument("--max-instances", type=int, default=99)
    p.add_argument("--warmup", type=float, default=60.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--hpgd-weight", type=float, default=0.5)
    p.add_argument("--device", choices=["cpu", "cuda", "gpu"], default="cpu")
    p.add_argument("--output", default="act/pipeline/Shiyang/results/_two_seg")
    a = p.parse_args()

    specs = VNNLibSpecCreator().create_specs_for_data_model_pairs(
        categories=[a.category], max_instances=a.max_instances)
    for model_id, wm in synthesize_models_from_specs(specs).items():
        seeds = initial_seeds_from_wrapped_model(wm)
        if not seeds:
            continue
        config = FuzzingConfig.from_yaml(
            max_iterations=10_000_000, timeout_seconds=a.warmup,
            seed_selection_strategy="energy",
            mutation_weights=build_mutation_weights(0.0, a.hpgd_weight),
            coverage_strategy="GlobalCov", activation_threshold=0.1,
            perturb_mode="adaptive_perdim", perturb_scale=1.0,
            save_counterexamples=False, output_dir=Path(a.output),
            report_interval=100000, verbose=0, trace_level=0,
            trace_sample_rate=1, trace_storage="json", trace_output=None,
            stop_on_first_violation=False,
            admission_mode="state", scheduling_mode="energy",
            enable_bi_gce=False, bi_threaded=False, bi_attack_strategy=None,
            state_bins=3, state_bin_tau=a.tau,
        )
        fuzzer = ACTFuzzer(wrapped_model=wm, initial_seeds=seeds, config=config)
        print(f"\n=== {model_id!r}: {a.warmup}s warmup, tau={a.tau}")
        fuzzer.fuzz()
        analyse(fuzzer, a)


if __name__ == "__main__":
    main()
