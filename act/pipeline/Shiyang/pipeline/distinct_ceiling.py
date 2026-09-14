"""How many of a group's instances are falsifiable AT ALL, given equal budget?

`distinct_instances_hit` from a fuzzing run confounds two things: how many
instances the search reached, and how many are breakable in the first place.
A campaign that reports distinct=1 has said nothing until you know whether the
ceiling is 1 (nothing to win) or 15 (lane starvation).

This measures the ceiling the only way that is fair to every instance: every
lane attacks its OWN spec row, every round, with a fresh random restart and
plain gradient ascent on the spec's own severity -- no corpus, no scheduling,
no lane competition. A lane counts as broken the first time it violates.

    python -m act.pipeline.Shiyang.pipeline.distinct_ceiling \
        --category eran_sigmoid_tanh_mlp --max-instances 30 --rounds 200
"""

from __future__ import annotations

import argparse
import time

import torch

from act.front_end.model_synthesis import synthesize_models_from_specs
from act.front_end.vnnlib_loader.create_specs import VNNLibSpecCreator


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--category", default="eran_sigmoid_tanh_mlp")
    p.add_argument("--max-instances", type=int, default=30)
    p.add_argument("--instance-range", type=int, nargs=2, default=None,
                   metavar=("START", "END"),
                   help="use instance indices [START, END) instead of the first N "
                        "(the tanh models start at 297 in eran_sigmoid_tanh_mlp)")
    p.add_argument("--model-index", type=int, default=0)
    p.add_argument("--rounds", type=int, default=200, help="random restarts")
    p.add_argument("--steps", type=int, default=30, help="ascent steps per restart")
    p.add_argument("--timeout", type=float, default=300.0)
    args = p.parse_args()

    creator = VNNLibSpecCreator()
    if args.instance_range:
        lo, hi = args.instance_range
        spec_results = creator.create_specs_for_data_model_pairs(
            categories=[args.category], instance_indices=list(range(lo, hi))
        )
    else:
        spec_results = creator.create_specs_for_data_model_pairs(
            categories=[args.category], max_instances=args.max_instances
        )
    models = list(synthesize_models_from_specs(spec_results).items())
    model_id, model = models[args.model_index]
    print(f"=== {model_id}")

    in_spec = next(m for m in model.modules() if type(m).__name__ == "InputSpecLayer").spec
    out_spec = next(m for m in model.modules() if type(m).__name__ == "OutputSpecLayer").spec
    lb, ub = in_spec.lb, in_spec.ub
    B = lb.shape[0]
    rows = torch.arange(B)
    step = float((ub - lb).max()) / max(args.steps, 1)

    broken = torch.zeros(B, dtype=torch.bool)
    best_sev = torch.full((B,), -float("inf"))
    t0 = time.time()
    for r in range(args.rounds):
        if time.time() - t0 > args.timeout:
            print(f"  [timeout after {r} rounds]")
            break
        x = lb + torch.rand_like(lb) * (ub - lb)
        for _ in range(args.steps):
            xr = x.detach().clone().requires_grad_(True)
            out = model(xr)
            outputs = out["output"] if isinstance(out, dict) else out
            _mask, sev = out_spec.violation(outputs, rows=rows)
            g = torch.autograd.grad(sev.sum(), xr)[0].detach()
            x = torch.max(torch.min(xr.detach() + step * g.sign(), ub), lb)
        with torch.no_grad():
            out = model(x)
            outputs = out["output"] if isinstance(out, dict) else out
            mask, sev = out_spec.violation(outputs, rows=rows)
        broken |= mask.detach().cpu().reshape(-1).bool()
        best_sev = torch.maximum(best_sev, sev.detach().cpu().reshape(-1))
        if (r + 1) % 25 == 0:
            print(f"  round {r + 1}: broken {int(broken.sum())}/{B}")

    hit = broken.nonzero(as_tuple=True)[0].tolist()
    print(f"  CEILING: {len(hit)}/{B} instances falsifiable -> rows {hit}")
    near = (~broken).nonzero(as_tuple=True)[0]
    if near.numel():
        order = torch.argsort(best_sev[near], descending=True)[:8]
        print("  closest unbroken rows (best severity, <0 = still safe): "
              + ", ".join(f"{int(near[i])}:{float(best_sev[near[i]]):.4f}" for i in order))


if __name__ == "__main__":
    main()
