"""Measure what a "state" could even mean on Sigmoid/Tanh networks.

The ReLU state is sign(z) over the UNSTABLE neurons: unstable = the box does
not pin the neuron's side of the kink.  Smooth activations have no kink, so
before designing a replacement partition we need to know which candidate
notions actually MOVE inside a verification box.  This reports, per activation
layer:

  bounds mode:
    cross-zero      neurons whose pre-activation interval straddles 0
                    (the literal port of "unstable"), from ACT's own interval
                    propagation -- i.e. the SOUND but loose count
    identity-band   max(|lo|,|hi|) <= 0.25, the case cons_exportor's
                    _emit_tanh_canonical treats as z ~ y

  sample mode (uniform inside the box, or its corners -- what a sign-gradient
  attack actually reaches):
    sign-volatile   neurons whose sign(z) is not constant over the samples
                    (the TIGHT count the loose one is meant to approximate)
    fixed+-1        neurons that change bin under fixed thresholds z = +-1
    quantile-bin    neurons that change bin under 3 per-neuron equal-frequency
                    bins (volatile by construction -- a sanity floor, not a
                    result)
    act-range       neurons whose OUTPUT sigma(z) spans more than 1e-3 / 1e-2 /
                    1e-1 over the samples: how much the neuron can actually say
                    inside this box, and the smooth analogue of "unstable"

Usage (from the repo root):
    python -m act.pipeline.Shiyang.pipeline.smooth_state_probe --mode corner
    python -m act.pipeline.Shiyang.pipeline.smooth_state_probe --mode bounds \
        --instances 0 297
"""

from __future__ import annotations

import argparse

import torch

from act.back_end.analyze import analyze
from act.back_end.core import Bounds, ConSet, Fact
from act.back_end.layer_schema import LayerKind
from act.back_end.verifier import find_entry_layer_id
from act.front_end.model_synthesis import synthesize_models_from_specs
from act.front_end.vnnlib_loader.create_specs import VNNLibSpecCreator
from act.pipeline.verification.torch2act import TorchToACT

# onnx2torch emits nn.Sigmoid for Sigmoid but an OnnxFunction wrapper for Tanh,
# so the fuzzing side cannot find these layers with an isinstance(nn.Tanh) test
# the way _relu_preactivations_batched finds nn.ReLU.  ReLU is included so the
# same probes can be pointed at a ReLU net (mnist_fc) as the reference point.
ACT_MODULE_TYPES = ("Sigmoid", "Tanh", "OnnxFunction", "ReLU")
ACT_LAYER_KINDS = (LayerKind.TANH.value, LayerKind.SIGMOID.value, LayerKind.RELU.value)


def activation_of(model_name: str):
    """The scalar activation these pre-activations feed, by model name."""
    up = model_name.upper()
    if "SIGMOID" in up:
        return torch.sigmoid
    if "TANH" in up:
        return torch.tanh
    return torch.relu


def smooth_preactivations(model, x, keep_graph: bool = False):
    """Per-layer pre-activations at every smooth-activation module, in order.

    keep_graph=True keeps the autograd graph, which is what the hinge
    projection needs (the same thing _relu_preactivations_batched does for
    HPGD on ReLU nets).
    """
    out, handles = [], []

    def hook(_m, inputs):
        if inputs and torch.is_tensor(inputs[0]):
            out.append(inputs[0] if keep_graph else inputs[0].detach())

    for m in model.modules():
        if type(m).__name__ in ACT_MODULE_TYPES:
            handles.append(m.register_forward_pre_hook(hook))
    try:
        model(x)
    finally:
        for h in handles:
            h.remove()
    return out


def flat_preactivations(model, x, keep_graph: bool = False):
    """[B, N] concatenated across layers, HPGD's coordinate system."""
    zs = smooth_preactivations(model, x, keep_graph=keep_graph)
    return torch.cat([z.flatten(start_dim=1) for z in zs], dim=1)


def load_instance(category: str, index: int):
    spec_results = VNNLibSpecCreator().create_specs_for_data_model_pairs(
        categories=[category], instance_indices=[index]
    )
    models = list(synthesize_models_from_specs(spec_results).items())
    if not models:
        raise RuntimeError(f"no wrapped model for {category} instance {index}")
    return models[0]


def report_bounds(model) -> None:
    spec = next(m for m in model.modules() if type(m).__name__ == "InputSpecLayer").spec
    net = TorchToACT(model).run()
    fact = Fact(bounds=Bounds(spec.lb.clone(), spec.ub.clone()), cons=ConSet())
    before, _after, _g = analyze(net, find_entry_layer_id(net), fact)
    for layer in net.layers:
        kind = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
        if kind not in ACT_LAYER_KINDS:
            continue
        b = before[layer.id].bounds
        lo, hi = b.lb.flatten(), b.ub.flatten()
        width = hi - lo
        print(f"  {kind} layer {layer.id}: n={lo.numel()} "
              f"z in [{lo.min():.2f},{hi.max():.2f}] "
              f"width mean={width.mean():.3f} max={width.max():.3f} | "
              f"cross-zero={((lo < 0) & (hi > 0)).sum().item()} "
              f"identity-band={(torch.maximum(lo.abs(), hi.abs()) <= 0.25).sum().item()}")


def report_samples(model, model_name: str, mode: str, n: int) -> None:
    spec = next(m for m in model.modules() if type(m).__name__ == "InputSpecLayer").spec
    lb, ub = spec.lb, spec.ub
    if mode == "uniform":
        x = lb + torch.rand(n, *lb.shape[1:]) * (ub - lb)
    else:
        mid, half = (lb + ub) / 2, (ub - lb) / 2
        x = mid + half * torch.where(torch.rand(n, *lb.shape[1:]) < 0.5, -1.0, 1.0)

    f = activation_of(model_name)
    with torch.no_grad():
        zs = smooth_preactivations(model, x)
    for li, z in enumerate(zs):
        z = z.flatten(start_dim=1)
        sign = z > 0
        sign_volatile = (sign.any(0) & (~sign).any(0)).sum().item()
        q = torch.quantile(z, torch.tensor([1 / 3, 2 / 3]), dim=0)
        digit = (z > q[0]).long() + (z > q[1]).long()
        quant_volatile = (digit.max(0).values != digit.min(0).values).sum().item()
        fixed = (z > -1.0).long() + (z > 1.0).long()
        fixed_volatile = (fixed.max(0).values != fixed.min(0).values).sum().item()
        act = f(z)
        rng = act.max(0).values - act.min(0).values
        live = [int((rng > t).sum()) for t in (1e-3, 1e-2, 1e-1)]
        print(f"  layer{li}: n={z.shape[1]} |z| mean={z.abs().mean():.2f} "
              f"| sign-volatile={sign_volatile} fixed+-1={fixed_volatile} "
              f"quantile-bin={quant_volatile} "
              f"| act-range>1e-3/1e-2/1e-1: {live[0]}/{live[1]}/{live[2]}")


def report_flip(model, model_name: str, k: int, steps: int, margin: float,
                rank: str = "normalized") -> None:
    """Take the k NEAREST sign states and actually try to reach them.

    Sampling can only say "a random walk does not move the state"; HPGD is
    targeted, so the honest test is to aim at the closest states and see
    whether the hinge projection lands them.  Nearest = smallest |z| at the
    seed, i.e. the least displacement that would change the state.

    Two numbers per instance:
      first-order reachable  |z0| < eps * ||dz/dx||_1 AT THE SEED, the box's
                             linear budget for moving that pre-activation.
                             Only an estimate, not a bound: the gradient
                             changes along the path, and measured, some
                             neurons flip that this test calls unreachable.
      actually flipped       sign really changed after `steps` of the
                             production hinge projection, one target neuron
                             per lane (the most favourable case: no competing
                             hold-still terms, no joint feasibility problem).
    """
    spec = next(m for m in model.modules() if type(m).__name__ == "InputSpecLayer").spec
    lb, ub = spec.lb, spec.ub
    x0 = (lb + ub) / 2
    half = (ub - lb) / 2

    z0 = flat_preactivations(model, x0)[0]
    n = z0.numel()

    # Full budget row ||dz_i/dx||_1 weighted by the box half-width, for EVERY
    # neuron, in one backward: replicate the seed into n lanes and take the
    # diagonal z_i of lane i, so lane i's input gradient is dz_i/dx.
    xr = x0.repeat(n, *([1] * (x0.dim() - 1))).clone().requires_grad_(True)
    z_rep = flat_preactivations(model, xr, keep_graph=True)
    diag = z_rep[torch.arange(n), torch.arange(n)]
    g = torch.autograd.grad(diag.sum(), xr)[0].detach()
    budget_all = (g.abs() * half).flatten(1).sum(1)          # [n]

    # Ranking the single-bit-flip neighbours. "absz" ranks by |z0|, which is a
    # distance in PRE-ACTIVATION space; "normalized" divides by the box's own
    # budget for moving that pre-activation, which is the distance in INPUT
    # space -- the metric the attack actually pays in. A steep neuron with a
    # large |z0| can be nearer than a flat one sitting on the boundary.
    if rank == "absz":
        order = torch.argsort(z0.abs())
    else:
        order = torch.argsort(z0.abs() / budget_all.clamp(min=1e-12))
    cand = order[:k]

    reach = budget_all[cand]
    margin0 = z0[cand].abs()
    first_order = (reach > margin0)

    # the production hinge projection, k lanes, one target neuron each
    x = x0.repeat(k, *([1] * (x0.dim() - 1))).clone()
    x_low, x_high = (x0 - half).expand_as(x), (x0 + half).expand_as(x)
    target = -torch.sign(z0[cand])
    step = float((2 * half).max()) / max(steps, 1)
    lanes = torch.arange(k)
    best = torch.full((k,), -float("inf"))
    zero_grad = 0
    for _ in range(steps):
        xr = x.detach().clone().requires_grad_(True)
        z = flat_preactivations(model, xr, keep_graph=True)
        zc = z[lanes, cand]
        best = torch.maximum(best, (target * zc).detach())
        violation = (margin - target * zc).clamp(min=0)
        loss = violation.sum()
        grad = torch.autograd.grad(loss, xr)[0].detach()
        # Only a lane that STILL wants to move and has no gradient is stuck;
        # a lane that already cleared the margin has zero gradient by right.
        zero_grad = int(((grad.flatten(1).abs().sum(1) == 0)
                         & (violation.detach() > 0)).sum())
        x = torch.max(torch.min(xr.detach() - step * grad.sign(), x_high), x_low)

    z_end = flat_preactivations(model, x)[lanes, cand]
    flipped = (torch.sign(z_end) == target)
    moved = (z_end - z0[cand]).abs()

    frac = (z0.abs() / budget_all.clamp(min=1e-12))
    print(f"  nearest {k} of {n} neurons by rank={rank}, {steps} hinge steps")
    print(f"    normalized margin |z0|/budget over ALL neurons: "
          f"<1: {int((frac < 1).sum())}  <2: {int((frac < 2).sum())}  "
          f"<5: {int((frac < 5).sum())}")
    print(f"    |z0| of those: min={margin0.min():.4f} median={margin0.median():.4f} "
          f"max={margin0.max():.4f}")
    print(f"    first-order reachable (eps*||dz/dx||_1 > |z0|): {int(first_order.sum())}/{k}")
    print(f"    ACTUALLY FLIPPED: {int(flipped.sum())}/{k}"
          f"   (of the first-order reachable: "
          f"{int((flipped & first_order).sum())}/{int(first_order.sum())})")
    print(f"    |dz| achieved: median={moved.median():.4f} max={moved.max():.4f} "
          f"| first-order budget median={reach.median():.4f}")
    print(f"    best signed margin toward target: max={best.max():.4f} "
          f"(>0 means some lane crossed) | STUCK lanes (want to move, zero grad)={zero_grad}")


def hinge_flip(model, x0, half, targets_idx, steps: int, margin: float):
    """Run the one-target-per-lane hinge projection; return (flipped, z_end)."""
    k = targets_idx.numel()
    z0 = flat_preactivations(model, x0)[0]
    x = x0.repeat(k, *([1] * (x0.dim() - 1))).clone()
    x_low, x_high = (x0 - half).expand_as(x), (x0 + half).expand_as(x)
    target = -torch.sign(z0[targets_idx])
    step = float((2 * half).max()) / max(steps, 1)
    lanes = torch.arange(k)
    for _ in range(steps):
        xr = x.detach().clone().requires_grad_(True)
        zc = flat_preactivations(model, xr, keep_graph=True)[lanes, targets_idx]
        loss = (margin - target * zc).clamp(min=0).sum()
        grad = torch.autograd.grad(loss, xr)[0].detach()
        x = torch.max(torch.min(xr.detach() - step * grad.sign(), x_high), x_low)
    z_end = flat_preactivations(model, x)[lanes, targets_idx]
    return (torch.sign(z_end) == target), z_end


def report_cells(model, model_name: str, show: int, steps: int, margin: float) -> None:
    """Spell out the partition on individual neurons: which cell each one is
    in, how wide that cell is, and whether the box can leave it.

    The partition is the ReLU one ported verbatim: two cells per neuron, split
    at the inflection point z = 0.  What differs from ReLU is that leaving a
    cell is not a qualitative change in the neuron's behaviour -- sigma(z) is
    smooth across the boundary -- so the table also prints what the neuron's
    OUTPUT actually does over the reachable range.
    """
    spec = next(m for m in model.modules() if type(m).__name__ == "InputSpecLayer").spec
    lb, ub = spec.lb, spec.ub
    x0, half = (lb + ub) / 2, (ub - lb) / 2
    f = activation_of(model_name)

    z0 = flat_preactivations(model, x0)[0]
    n = z0.numel()
    xr = x0.repeat(n, *([1] * (x0.dim() - 1))).clone().requires_grad_(True)
    z_rep = flat_preactivations(model, xr, keep_graph=True)
    diag = z_rep[torch.arange(n), torch.arange(n)]
    g = torch.autograd.grad(diag.sum(), xr)[0].detach()
    budget = (g.abs() * half).flatten(1).sum(1)

    # The SOUND bounds, for contrast: the gradient budget above is a local
    # first-order estimate at one point, IBP is what the verifier actually
    # propagates over the whole box.
    net = TorchToACT(model).run()
    fact = Fact(bounds=Bounds(lb.clone(), ub.clone()), cons=ConSet())
    before, _after, _g2 = analyze(net, find_entry_layer_id(net), fact)
    ibp_lo, ibp_hi = [], []
    for layer in net.layers:
        kind = layer.kind.upper() if isinstance(layer.kind, str) else layer.kind
        if kind not in ACT_LAYER_KINDS:
            continue
        b = before[layer.id].bounds
        ibp_lo.append(b.lb.flatten())
        ibp_hi.append(b.ub.flatten())
    ibp_lo, ibp_hi = torch.cat(ibp_lo), torch.cat(ibp_hi)
    if ibp_lo.numel() != n:
        raise RuntimeError(f"IBP gave {ibp_lo.numel()} neurons, hooks gave {n}")

    order = torch.argsort(z0.abs() / budget.clamp(min=1e-12))
    per_layer = [z.shape[1] for z in smooth_preactivations(model, x0)]
    probe = torch.cat([order[:show], order[-show:]])
    flipped, _z_end = hinge_flip(model, x0, half, probe, steps, margin)

    def layer_of(i: int):
        acc = 0
        for li, w in enumerate(per_layer):
            if i < acc + w:
                return li, i - acc
            acc += w
        return -1, i

    print(f"  {n} neurons, 2 cells each (z<0 | z>=0), {f.__name__}")
    print(f"  {'neuron':>10} {'z0':>9} {'cell':>4} {'budget':>7} "
          f"{'grad-estimate z':>18} {'IBP bounds (sound)':>22} "
          f"{'IBP width':>10}  flips?")
    for rowset, title in ((probe[:show], "NEAREST"), (probe[show:], "FARTHEST")):
        print(f"  -- {title} {show} by |z0|/budget --")
        for j, i in enumerate(rowset):
            i = int(i)
            li, ii = layer_of(i)
            lo, hi = float(z0[i] - budget[i]), float(z0[i] + budget[i])
            _ = f  # activation range moved out; the bounds contrast is the point
            did = bool(flipped[j if title == "NEAREST" else show + j])
            bl, bh = float(ibp_lo[i]), float(ibp_hi[i])
            print(f"  L{li}[{ii:>3}] {float(z0[i]):>9.4f} "
                  f"{'+' if z0[i] >= 0 else '-':>4} {float(budget[i]):>7.4f} "
                  f"[{lo:>7.3f},{hi:>7.3f}] [{bl:>9.3f},{bh:>9.3f}] "
                  f"{bh - bl:>10.3f}  {'YES' if did else 'no'}")


def confirmed_flippable(model, x0, half, steps: int, margin: float,
                        chunk: int = 200) -> torch.Tensor:
    """Indices of every neuron a single-target hinge projection can actually
    flip -- the real state space, as opposed to what IBP calls unstable."""
    n = flat_preactivations(model, x0).shape[1]
    hits = []
    for start in range(0, n, chunk):
        idx = torch.arange(start, min(start + chunk, n))
        flipped, _ = hinge_flip(model, x0, half, idx, steps, margin)
        hits.append(idx[flipped])
    return torch.cat(hits)


def report_hpgd(model, model_name: str, ks, trials: int, steps: int,
                margin: float) -> None:
    """Production HPGD (flip K neurons at once) restricted to the state space
    that demonstrably exists.

    Two arms, matched on K and on step budget:
      confirmed  K neurons drawn from the set a single-target projection was
                 measured to flip -- every requested flip is individually
                 known to be reachable, so anything that fails is a JOINT
                 feasibility failure, not an unreachable target.
      all-neurons  K neurons drawn uniformly from every neuron, i.e. what HPGD
                 does when the candidate mask is wrong or absent.
    """
    spec = next(m for m in model.modules() if type(m).__name__ == "InputSpecLayer").spec
    lb, ub = spec.lb, spec.ub
    x0, half = (lb + ub) / 2, (ub - lb) / 2

    z0 = flat_preactivations(model, x0)[0]
    n = z0.numel()
    natural = torch.where(z0 > 0, 1.0, -1.0)
    cand = confirmed_flippable(model, x0, half, steps, margin)
    print(f"  confirmed-flippable state space: {cand.numel()}/{n} neurons")

    x_low, x_high = x0 - half, x0 + half
    step = float((2 * half).max()) / max(steps, 1)

    for arm, pool in (("confirmed", cand), ("all-neurons", torch.arange(n))):
        print(f"  -- arm={arm} (pool={pool.numel()}) --")
        for k in ks:
            if k > pool.numel():
                continue
            # one lane per trial; each lane asks for its own K flips
            asked = torch.stack([pool[torch.randperm(pool.numel())[:k]]
                                 for _ in range(trials)])           # [T, k]
            target = natural.unsqueeze(0).repeat(trials, 1)          # [T, n]
            rows = torch.arange(trials).unsqueeze(1).expand_as(asked)
            target[rows, asked] *= -1

            x = x0.repeat(trials, *([1] * (x0.dim() - 1))).clone()
            xl, xh = x_low.expand_as(x), x_high.expand_as(x)
            weight = torch.zeros(trials, n)
            weight[rows, asked] = 1.0
            for _ in range(steps):
                xr = x.detach().clone().requires_grad_(True)
                z = flat_preactivations(model, xr, keep_graph=True)
                loss = ((margin - target * z).clamp(min=0) * weight).sum()
                grad = torch.autograd.grad(loss, xr)[0].detach()
                x = torch.max(torch.min(xr.detach() - step * grad.sign(), xh), xl)

            z_end = flat_preactivations(model, x)
            sign_end = torch.where(z_end > 0, 1.0, -1.0)
            landed = (sign_end[rows, asked] == target[rows, asked]).float().sum(1)
            moved = (sign_end != natural.unsqueeze(0))
            collateral = moved.float().sum(1) - landed
            print(f"    K={k:>2}: landed {landed.mean():>5.2f}/{k} "
                  f"({landed.mean() / k:>5.1%})  all-asked {(landed == k).float().mean():>5.1%}  "
                  f"collateral flips {collateral.mean():>5.2f}")


def _project_to_target(model, x0, half, target, weight, steps, margin):
    """The production hinge projection toward a full sign-pattern target."""
    x = x0.repeat(target.shape[0], *([1] * (x0.dim() - 1))).clone()
    xl, xh = (x0 - half).expand_as(x), (x0 + half).expand_as(x)
    step = float((2 * half).max()) / max(steps, 1)
    for _ in range(steps):
        xr = x.detach().clone().requires_grad_(True)
        z = flat_preactivations(model, xr, keep_graph=True)
        loss = ((margin - target * z).clamp(min=0) * weight).sum()
        grad = torch.autograd.grad(loss, xr)[0].detach()
        x = torch.max(torch.min(xr.detach() - step * grad.sign(), xh), xl)
    return x.detach()


def report_reach(model, model_name: str, steps: int, margin: float,
                 per_bucket: int, seed_index: int = 0,
                 eps_scale: float = 1.0) -> None:
    """Can HPGD return to a state a real point was already observed in?

    Protocol: from the instance's own input x0, perturb inside the box to x1
    and record both sign patterns, A = s(x0) and B = s(x1).  B is then a
    target that is KNOWN reachable -- x1 realises it -- so a miss is HPGD
    failing to steer, not an infeasible target.  Perturbation magnitude is
    swept so targets land at a range of Hamming distances d = |A xor B|.

    Each real target is matched against two fabricated ones at the SAME d:
      synth-conf  d random flips drawn from the confirmed-flippable set
      synth-all   d random flips drawn from every neuron
    so the comparison controls displacement, not just hit count.
    """
    spec = next(m for m in model.modules() if type(m).__name__ == "InputSpecLayer").spec
    lb, ub = spec.lb, spec.ub
    input_layer = next(m for m in model.modules() if type(m).__name__ == "InputLayer")
    if eps_scale != 1.0:
        # Inflate the box around its own centre. This leaves the verification
        # property behind -- it is a probe of the network, not a query -- and
        # exists to separate "smooth activations have a small state space"
        # from "these instances just ship a smaller epsilon than mnist_fc".
        c, hw = (lb + ub) / 2, (ub - lb) / 2 * eps_scale
        lb, ub = (c - hw).clamp(0.0, 1.0), (c + hw).clamp(0.0, 1.0)
        print(f"  [eps x{eps_scale}: box half-width now {float(hw.max()):.4f}]")
    x_real = getattr(getattr(input_layer, "labeled_input", None), "tensor", None)
    x0 = (lb + ub) / 2 if x_real is None else torch.max(torch.min(
        x_real[seed_index:seed_index + 1].to(lb.dtype), ub), lb)
    half = (ub - lb) / 2

    zA = flat_preactivations(model, x0)[0]
    n = zA.numel()
    A = torch.where(zA > 0, 1.0, -1.0)
    pool = confirmed_flippable(model, x0, half, steps, margin)
    print(f"  seed from {'the real input' if x_real is not None else 'box centre'}"
          f", {n} neurons, confirmed-flippable {pool.numel()}")

    # Two sources of real perturbed points, because random directions barely
    # move the state: incoherent (scaled random corners) and coherent
    # (sign-gradient ascent on a RANDOM linear functional of z, which pushes
    # many neurons the same way at once). The coherent one names no target
    # pattern, so it is not circular -- it just walks to a genuine faraway
    # point whose state we then read off.
    alphas = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    xs = []
    for a in alphas:
        u = torch.where(torch.rand(per_bucket, *x0.shape[1:]) < 0.5, -1.0, 1.0)
        xs.append(torch.max(torch.min(x0 + a * half * u, ub), lb))
    for push_steps in (2, 5, 10, 25, 50):
        r = torch.randn(per_bucket, n)
        x = x0.repeat(per_bucket, *([1] * (x0.dim() - 1))).clone()
        xl, xh = (x0 - half).expand_as(x), (x0 + half).expand_as(x)
        st = float((2 * half).max()) / max(push_steps, 1)
        for _ in range(push_steps):
            xr = x.detach().clone().requires_grad_(True)
            z = flat_preactivations(model, xr, keep_graph=True)
            g = torch.autograd.grad((r * z).sum(), xr)[0].detach()
            x = torch.max(torch.min(xr.detach() + st * g.sign(), xh), xl)
        xs.append(x.detach())
    x1 = torch.cat(xs)
    with torch.no_grad():
        B_all = torch.where(flat_preactivations(model, x1) > 0, 1.0, -1.0)
    d_all = (B_all != A.unsqueeze(0)).sum(1)

    edges = [(1, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 10 ** 9)]
    print(f"  {'d (real)':>9} {'n':>4} | "
          f"{'real landed':>11} {'all-askd':>8} {'whole':>6} {'end-d':>6} | "
          f"{'conf landed':>11} {'all-askd':>8} {'whole':>6} | "
          f"{'all landed':>10} {'whole':>6}")
    for lo_d, hi_d in edges:
        sel = ((d_all >= lo_d) & (d_all <= hi_d)).nonzero(as_tuple=True)[0][:per_bucket]
        if sel.numel() == 0:
            continue
        m = sel.numel()
        rows = torch.arange(m)

        real_T = B_all[sel]
        real_W = (real_T != A.unsqueeze(0)).float()
        d = real_W.sum(1)

        def synth(src_pool):
            T = A.unsqueeze(0).repeat(m, 1).clone()
            for j in range(m):
                kk = min(int(d[j]), src_pool.numel())
                pick = src_pool[torch.randperm(src_pool.numel())[:kk]]
                T[j, pick] *= -1
            return T, (T != A.unsqueeze(0)).float()

        conf_T, conf_W = synth(pool)
        all_T, all_W = synth(torch.arange(n))

        out = {}
        for tag, (T, W) in (("real", (real_T, real_W)),
                            ("conf", (conf_T, conf_W)),
                            ("all", (all_T, all_W))):
            x_end = _project_to_target(model, x0, half, T, W, steps, margin)
            with torch.no_grad():
                s_end = torch.where(flat_preactivations(model, x_end) > 0, 1.0, -1.0)
            asked = W.bool()
            landed = ((s_end == T) & asked).float().sum(1)
            frac = (landed / W.sum(1).clamp(min=1))
            # two different bars, kept apart because they answer different
            # questions: did every REQUESTED flip land (ignoring collateral),
            # vs is the WHOLE pattern the target named now in place.
            all_asked = (landed == W.sum(1)).float().mean()
            whole = ((s_end == T).all(dim=1)).float().mean()
            end_dist = (s_end != T).float().sum(1).mean()
            out[tag] = (frac.mean(), all_asked, whole, end_dist)

        print(f"  {lo_d:>3}-{hi_d if hi_d < 10**9 else '':<5} {m:>4} | "
              f"{out['real'][0]:>11.1%} {out['real'][1]:>8.1%} "
              f"{out['real'][2]:>6.1%} {out['real'][3]:>6.1f} | "
              f"{out['conf'][0]:>11.1%} {out['conf'][1]:>8.1%} {out['conf'][2]:>6.1%} | "
              f"{out['all'][0]:>10.1%} {out['all'][2]:>6.1%}")
    print("  landed = share of the REQUESTED flips that landed; "
          "all-askd = every requested flip landed;")
    print("  whole  = the entire pattern equals the target "
          "(so any collateral flip fails it); end-d = bits still off.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--category", default="eran_sigmoid_tanh_mlp")
    p.add_argument("--instances", type=int, nargs="+", default=[0, 297],
                   help="instance indices (0 = sigmoid 6x100, 297 = tanh 6x100)")
    p.add_argument("--mode",
                   choices=["bounds", "uniform", "corner", "flip", "cells", "hpgd", "reach"],
                   default="corner")
    p.add_argument("--eps-scale", type=float, default=1.0,
                   help="reach mode: inflate the box by this factor (probe only, "
                        "leaves the verification property behind)")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 5, 10],
                   help="hpgd mode: simultaneous flip budgets to compare")
    p.add_argument("--trials", type=int, default=32,
                   help="hpgd mode: random target draws per K")
    p.add_argument("--show", type=int, default=8,
                   help="cells mode: how many neurons at each end of the table")
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--k", type=int, default=60, help="flip mode: how many nearest states")
    p.add_argument("--steps", type=int, default=50, help="flip mode: hinge steps")
    p.add_argument("--margin", type=float, default=0.01)
    p.add_argument("--rank", choices=["absz", "normalized"], default="normalized",
                   help="flip mode: which metric defines the NEAREST states")
    args = p.parse_args()

    for idx in args.instances:
        model_id, model = load_instance(args.category, idx)
        name = model_id[1] if isinstance(model_id, tuple) else str(model_id)
        print(f"=== {name} [{args.mode}]")
        if args.mode == "bounds":
            report_bounds(model)
        elif args.mode == "reach":
            report_reach(model, name, args.steps, args.margin, args.trials,
                         eps_scale=args.eps_scale)
        elif args.mode == "hpgd":
            report_hpgd(model, name, args.ks, args.trials, args.steps, args.margin)
        elif args.mode == "cells":
            report_cells(model, name, args.show, args.steps, args.margin)
        elif args.mode == "flip":
            report_flip(model, name, args.k, args.steps, args.margin, args.rank)
        else:
            report_samples(model, name, args.mode, args.samples)


if __name__ == "__main__":
    main()
