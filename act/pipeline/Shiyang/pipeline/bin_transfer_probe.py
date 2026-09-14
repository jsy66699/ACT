"""Can HPGD put a NAMED neuron in a NAMED segment?

Expressing the move is settled -- `StateBinning.write_bin` names any of the six
(source, destination) segment pairs exactly.  Reaching it is a different
question, and the two differ by a lot: high -> middle crosses one wall, high ->
low crosses two, and the second needs the box to move the pre-activation by
more than 2*tau rather than by however far it is from the nearest wall.

So this runs the PRODUCTION hinge projection, one target neuron per lane, and
reports per (source segment -> destination segment) how often the neuron
actually ends up in the segment it was told to.  It is the most favourable
setting available -- a single target, no competing hold-still terms, no joint
feasibility problem across k neurons -- so the rates here are an upper bound on
what the k=10 production path can do, not a prediction of it.

    python -m act.pipeline.Shiyang.pipeline.bin_transfer_probe
    python -m act.pipeline.Shiyang.pipeline.bin_transfer_probe --tau 0.5 --steps 100
"""

from __future__ import annotations

import argparse

import torch

from act.pipeline.fuzzing.state_bins import StateBinning
from act.pipeline.Shiyang.pipeline.smooth_state_probe import (
    flat_preactivations, load_instance,
)

SEG = {0: "low", 1: "mid", 2: "high"}


def probe(model, binning: StateBinning, per_pair: int, steps: int, margin: float,
          step_mult: float = 1.0, box_mult: float = 1.0):
    """{(src, dst): (landed, tried, median |dz|, median wall distance)}."""
    spec = next(m for m in model.modules() if type(m).__name__ == "InputSpecLayer").spec
    lb, ub = spec.lb, spec.ub
    x0, half = (lb + ub) / 2, (ub - lb) / 2 * box_mult

    z0 = flat_preactivations(model, x0)[0]
    n = z0.numel()
    code0 = binning.code(z0.unsqueeze(0))
    bins0 = binning.bin_of(code0)[0]

    # The box's first-order budget for moving each pre-activation, so a failure
    # can be read as "no room" rather than "the projection is broken".
    xr = x0.repeat(n, *([1] * (x0.dim() - 1))).clone().requires_grad_(True)
    z_rep = flat_preactivations(model, xr, keep_graph=True)
    diag = z_rep[torch.arange(n), torch.arange(n)]
    g = torch.autograd.grad(diag.sum(), xr)[0].detach()
    budget = (g.abs() * half).flatten(1).sum(1)

    out = {}
    for src in (0, 1, 2):
        pool = (bins0 == src).nonzero(as_tuple=True)[0]
        if pool.numel() == 0:
            continue
        # Rank by how far the neuron is from the wall it must cross, relative
        # to what the box can pay -- the same "nearest states first" ordering
        # the sign-flip probe uses, so this is not a random sample of neurons.
        for dst in (0, 1, 2):
            if dst == src:
                continue
            # The FIRST wall on the way. Segment boundaries are -tau (low|mid)
            # and +tau (mid|high), so going up from low or down from mid both
            # cross -tau, and everything else crosses +tau first. A two-segment
            # move crosses that one and then the other.
            going_up = dst > src
            wall = -binning.tau if (src == 0 or (src == 1 and not going_up)) \
                else binning.tau
            need = (z0[pool] - wall).abs()
            order = torch.argsort(need / budget[pool].clamp(min=1e-12))
            cand = pool[order[:per_pair]]
            k = cand.numel()
            if k == 0:
                continue

            target = code0.repeat(k, 1).clone()
            binning.write_bin(target, torch.arange(k), cand,
                              torch.full((k,), dst, dtype=torch.long))
            # Score ONLY the target neuron's two coordinates.
            weight = torch.zeros_like(target)
            weight[torch.arange(k), 2 * cand] = 1.0
            weight[torch.arange(k), 2 * cand + 1] = 1.0

            x = x0.repeat(k, *([1] * (x0.dim() - 1))).clone()
            x_low, x_high = (x0 - half).expand_as(x), (x0 + half).expand_as(x)
            step = float((2 * half).max()) / max(steps, 1) * step_mult
            for _ in range(steps):
                xr = x.detach().clone().requires_grad_(True)
                z = flat_preactivations(model, xr, keep_graph=True)
                viol = (margin - target * binning.shifted(z)).clamp(min=0)
                grad = torch.autograd.grad((viol * weight).sum(), xr)[0].detach()
                x = torch.max(torch.min(xr.detach() - step * grad.sign(), x_high), x_low)

            z_end = flat_preactivations(model, x)
            end_bin = binning.bin_of(binning.code(z_end))[torch.arange(k), cand]
            landed = end_bin == dst
            moved = (z_end[torch.arange(k), cand] - z0[cand]).abs()
            # Where the FAILURES stopped. For a two-segment move this is the
            # whole question: stalling in the middle segment is partial
            # progress (the code did change, and a second projection could
            # continue from there), while staying in the source segment means
            # the projection bought nothing at all.
            where = tuple(int((end_bin == s).sum()) for s in (0, 1, 2))
            out[(src, dst)] = (int(landed.sum()), k, float(moved.median()),
                               float(need[order[:k]].median()), where)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="eran_sigmoid_tanh_mlp")
    p.add_argument("--instances", type=int, nargs="+", default=[0, 297])
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--per-pair", type=int, default=40)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--margin", type=float, default=0.01)
    p.add_argument("--step-mult", type=float, default=1.0,
                   help="Scale the per-step size. 1.0 means `steps` steps sum "
                        "to exactly the box width, so the projection can already "
                        "reach any corner; >1 only makes it coarser, since the "
                        "box clamp is what actually bounds the displacement.")
    p.add_argument("--box-mult", type=float, default=1.0,
                   help="Scale the clamp box. >1 leaves the verification box, so "
                        "the result is NOT a valid counterexample for the "
                        "instance -- diagnostic only, to separate 'the box is "
                        "too small' from 'the projection is too weak'.")
    a = p.parse_args()

    for idx in a.instances:
        model_id, wm = load_instance(a.category, idx)
        name = str(model_id[1]) if isinstance(model_id, tuple) else str(model_id)
        z = flat_preactivations(wm, (next(
            m for m in wm.modules() if type(m).__name__ == "InputSpecLayer").spec.lb))
        binning = StateBinning.build(z.shape[1], bins=3, tau=a.tau)
        res = probe(wm, binning, a.per_pair, a.steps, a.margin,
                    step_mult=a.step_mult, box_mult=a.box_mult)
        print(f"=== {name}  tau={a.tau}  {a.steps} steps x{a.step_mult} "
              f"box x{a.box_mult}, {a.per_pair} nearest per pair")
        print(f"    {'move':<14}{'landed':>10}{'rate':>8}"
              f"{'median |dz|':>13}{'wall dist':>11}   ended in low/mid/high")
        for (src, dst), (ok, k, dz, need, where) in sorted(res.items()):
            print(f"    {SEG[src] + ' -> ' + SEG[dst]:<14}{f'{ok}/{k}':>10}"
                  f"{ok / max(k, 1) * 100:>7.0f}%{dz:>13.3f}{need:>11.3f}"
                  f"   {where[0]:>4} /{where[1]:>4} /{where[2]:>4}")


if __name__ == "__main__":
    main()
