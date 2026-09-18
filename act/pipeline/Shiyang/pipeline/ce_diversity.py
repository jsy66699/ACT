"""Counterexample-diversity (D, S) analysis, for any benchmark.

    python -m act.pipeline.Shiyang.pipeline.ce_diversity cora
    python -m act.pipeline.Shiyang.pipeline.ce_diversity safenlp
    python -m act.pipeline.Shiyang.pipeline.ce_diversity all      # cross-bench

Reads the `--dump-founder-clusters` dumps written under
`results/div_<bench>/<arm>/<group>/ce_sample.npz` and reports, per arm:

    D  instances broken                          (inter-instance diversity)
    S  mean pairwise cosine distance between the
       counterexamples' affine constraint directions (intra-instance)

The metric itself is derived in docs/CE_DIVERSITY_METRIC.md.

Generalised from the cora/mnist-only version in two places, both forced by
cifar100 and safenlp rather than chosen:

  * The constraint direction is now `d severity / d x`, taken from the group's
    own `OutputSpec`, instead of a hand-written TOP1 margin. safenlp's specs
    are `UNSAFE_LINEAR` and carry no true class at all (`true_class == -1` in
    every dump), so the TOP1 margin is not merely inconvenient there, it is
    undefined. `OutputSpec.severity` is the same signed margin the fuzzer
    ascends, and on TOP1_ROBUST it IS that expression, so the two benchmarks
    already measured are unaffected: re-running cora reproduced D
    47/126/105/111 and the same example instance, and its per-instance S
    values match the previous run to 1e-8 (largest gap 4.7e-6, float32 vs
    float64 rounding where the runner-up class is nearly tied).
  * Models come from ACT's own loader and are matched to result directories by
    recomputing the driver's `safe_name`, instead of parsing an ONNX filename
    out of the directory with a per-benchmark regex. cifar100's inputs are
    (3,32,32) and its wrapped model is a ResNet, so the old flatten-and-feed
    path could not have run there either.

S is exact: every dumped counterexample, all pairs, no rarefaction.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "act/pipeline/Shiyang/results"

# Two arm sets share this script.
#
# "energy" is the original four: the CE-energy / CE-parent-replacement
# scheduling line, with state admission and HPGD stacked ON TOP of it.  Its
# state and hpgd arms therefore do NOT isolate admission or guidance -- each
# also carries ce_energy 1 + cerepl, so they cannot be read as "what state
# admission is worth" once the scheduling line is out of scope.
#
# "portfolio" is the three arms that vary ONLY admission and the mutation
# portfolio, with every energy switch left at its published default.
ARM_SETS = {
    "energy": ["baseline", "ce1_cerepl", "ce1_cerepl_state", "hpgd_ce1_cerepl"],
    "portfolio": ["baseline", "statebase", "statehpgd"],
    # Dropping statebase is not only a narrower figure: the paired set is an
    # INTERSECTION, so requiring two arms to have broken an instance instead of
    # three admits more of them, and the paired S rests on more instances.
    "guided": ["baseline", "statehpgd"],
    # 2-bin vs 3-bin state code, paired within each admission arm. There is no
    # `baseline` here on purpose: the question is what the PARTITION changes,
    # so both members of each pair must differ only in --state-bins.
    "bins": ["statebase_b2", "statebase_b3", "statehpgd_b2", "statehpgd_b3"],
}
ARMS = ARM_SETS["energy"]
TAG = "4arm"
# The panel titles state a finding, so they belong to the arm set.  The
# four-arm wording names a scheduling arm the portfolio set does not have,
# and its D ranking came from a campaign; the portfolio D below is one run
# per arm and does NOT reproduce the campaign means (safenlp statebase 36.60
# vs statehpgd 32.20 over n=5 lands 30 vs 37 in a single run).
# One title per panel: D, paired S, unpaired S.  The titles state a finding,
# so they belong to the arm set -- and panels 2 and 3 genuinely disagree, so
# neither may borrow the other's wording.
PANEL_TITLES = {
    "energy": (
        "Inter-instance: D, one run per arm\n"
        "-- scheduling wins, and HPGD does not add to it",
        "Intra-instance, PAIRED: HPGD is widest on every benchmark\n"
        "-- the opposite ranking to the left panel",
        "Intra-instance, UNPAIRED: the ranking changes\n"
        "-- HPGD is widest on 2 of 4",
    ),
    "portfolio": (
        "Inter-instance: D, campaign mean +- sd\n"
        "-- state admission wins; HPGD adds only on mnist",
        "Intra-instance, PAIRED: HPGD is widest on all 4\n"
        "-- cora is 20 of 20 instances; the rest on 1-3",
        "Intra-instance, UNPAIRED: only cora separates the arms\n"
        "-- mnist reverses, but every bar overlaps",
    ),
    "guided": (
        "Inter-instance: D, campaign mean +- sd\n"
        "-- statehpgd breaks more on every benchmark",
        "Intra-instance, PAIRED: statehpgd is wider on all 4\n"
        "-- cora 26 of 27 instances, safenlp 5 of 5",
        "Intra-instance, UNPAIRED: cora and safenlp separate\n"
        "-- mnist reverses, but its error bars overlap",
    ),
}
ARM_SET_NAME = "energy"
# Per-benchmark example-instance overrides, as {bench: (group, instance)}.
# Empty by default: the effect-blind rule in figures() picks the example.
EXAMPLE_OVERRIDE = {}
TITLES = {"baseline": "baseline\n(paper portfolio)",
          "ce1_cerepl": "ce1_cerepl\n(CE-energy 1 + CE-parent-replace)",
          "ce1_cerepl_state": "state_ce1_cerepl\n(+ state admission)",
          "hpgd_ce1_cerepl": "hpgd_ce1_cerepl\n(+ HPGD 0.5)",
          "statebase": "statebase\n(+ state admission)",
          "statehpgd": "statehpgd\n(+ state admission + HPGD 0.5)"}
SHORT = {"baseline": "baseline", "ce1_cerepl": "ce1_cerepl",
         "ce1_cerepl_state": "state_ce1_cerepl", "hpgd_ce1_cerepl": "hpgd_ce1_cerepl",
         "statebase": "statebase", "statehpgd": "statehpgd",
         "statebase_b2": "state 2-bin", "statebase_b3": "state 3-bin",
         "statehpgd_b2": "hpgd 2-bin", "statehpgd_b3": "hpgd 3-bin"}
# categorical slots 7/1/2/3; validated all-pairs (normal-vision dE 16.3, CVD 9.2)
COL = {"baseline": "#4a3aa7", "ce1_cerepl": "#2a78d6",
       "ce1_cerepl_state": "#eb6834", "hpgd_ce1_cerepl": "#1baf7a",
       "statebase": "#eb6834", "statehpgd": "#1baf7a",
       # Paired hues: the two admission arms keep their colours from the
       # portfolio figure, and the 3-bin member of each pair is the darker one.
       "statebase_b2": "#eb6834", "statebase_b3": "#8c3a12",
       "statehpgd_b2": "#1baf7a", "statehpgd_b3": "#0d6547"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8880", "#d8d7d2"

BENCH = {
    "cora": dict(dir="div_cora", category="cora_2024", max_instances=180,
                 group_re=r"^([a-z0-9]+-[a-z]+)",
                 label="cora_2024, 180 specs / 9 groups / 60 s per group"),
    "mnist": dict(dir="div_mnist", category="mnist_fc_v2", max_instances=90,
                  group_re=r"^(mnist-net_256x[0-9]+)",
                  label="mnist_fc_v2, 90 specs / 3 groups / 60 s per group"),
    "cifar": dict(dir="div_cifar", category="cifar100_2024", max_instances=200,
                  group_re=r"^CIFAR100_(resnet_(?:large|medium))",
                  label="cifar100_2024, 200 specs / 2 groups / 60 s per group"),
    "safenlp": dict(dir="div_safenlp", category="safenlp_2024", max_instances=200,
                    group_re=r"(hyperrectangle_[0-9]+)",
                    label="safenlp_2024, 200 specs / 2 groups / 60 s per group"),
    # One model group holds all 200 specs, so a trial yields one row and n is
    # the repeat count, not repeats x groups as everywhere else.
    "tin": dict(dir="div_tin", category="tinyimagenet_v2", max_instances=200,
                group_re=r"^TinyImageNet_(resnet_(?:medium|large))",
                label="tinyimagenet_v2, 200 specs / 1 group / 60 s"),
    # The two smooth-activation entries. Unlike the four above, each dump run
    # covered ONE model group (--instance-indices), because the category holds
    # six ONNX nets and a "first N specs" prefix cannot isolate the tanh ones:
    # tanh 6x100 starts at row 297, so max_instances has to reach it and the
    # three sigmoid groups come along unused (analyse() skips a group with no
    # ce_sample.npz).
    "eransig": dict(dir="div_eran_sig6x100", category="eran_sigmoid_tanh_mlp",
                    max_instances=99,
                    group_re=r"^(ffnn(?:SIGMOID|TANH)__Point_[0-9]+x[0-9]+)",
                    label="eran sigmoid 6x100, 99 specs / 1 group / 60 s"),
    "erantanh": dict(dir="div_eran_tanh6x100", category="eran_sigmoid_tanh_mlp",
                     max_instances=394,
                     group_re=r"^(ffnn(?:SIGMOID|TANH)__Point_[0-9]+x[0-9]+)",
                     label="eran tanh 6x100, 97 specs / 1 group / 60 s"),
    "bins3sig": dict(dir="div_bins3_sig6x100", category="eran_sigmoid_tanh_mlp",
                     max_instances=99,
                     group_re=r"^(ffnn(?:SIGMOID|TANH)__Point_[0-9]+x[0-9]+)",
                     label="eran sigmoid 6x100, 2-bin vs 3-bin state"),
    "bins3tanh": dict(dir="div_bins3_tanh6x100", category="eran_sigmoid_tanh_mlp",
                      max_instances=394,
                      group_re=r"^(ffnn(?:SIGMOID|TANH)__Point_[0-9]+x[0-9]+)",
                      label="eran tanh 6x100, 2-bin vs 3-bin state"),
}
# Campaign distinct means -- (mean, sd, n) -- the number the paper cites.
# A single diversity run cannot rank arms (see combined.__doc__), so the D axis
# reads from here.  cora_2024 is absent on purpose: it never had a distinct
# campaign, only the CE-diversity dumps, and a fabricated entry would be worse
# than an honest n=1 marker.
CAMPAIGN_D = {
    "cifar": {"baseline": (3.57, 1.07, 30), "statebase": (6.77, 1.79, 30),
              "statehpgd": (6.37, 2.33, 30)},
    "mnist": {"baseline": (4.67, 0.99, 30), "statebase": (5.53, 1.20, 30),
              "statehpgd": (6.83, 1.53, 30)},
    "safenlp": {"baseline": (11.00, 2.55, 5), "statebase": (36.60, 9.94, 5),
                "statehpgd": (32.20, 6.83, 5)},
    # 2026-09-09, n=5 x 60 s, `results/_eran_state_ab_{sig,tanh}6x100`.  The
    # smooth-activation pair: state admission is the only arm that moves
    # distinct here, and HPGD on top of it does not add (sigmoid) or subtracts
    # (tanh), which is the same ordering the ReLU campaigns show.
    "eransig": {"baseline": (1.6, 0.8, 5), "statebase": (4.6, 1.2, 5),
                "statehpgd": (4.4, 0.8, 5)},
    # 2026-09-10, n=5, `results/_eran_bins3_*`. Both b2 arms were re-run on the
    # state_bins code so the pair sits on one version.
    "bins3sig": {"statebase_b2": (3.6, 0.5, 5), "statebase_b3": (4.8, 0.7, 5),
                 "statehpgd_b2": (4.4, 1.2, 5), "statehpgd_b3": (4.4, 1.0, 5)},
    "bins3tanh": {"statebase_b2": (1.6, 0.5, 5), "statebase_b3": (2.2, 0.4, 5),
                  "statehpgd_b2": (1.6, 0.5, 5), "statehpgd_b3": (2.6, 0.5, 5)},
    "erantanh": {"baseline": (1.4, 0.5, 5), "statebase": (2.2, 0.4, 5),
                 "statehpgd": (1.8, 0.7, 5)},
}
ORDER = ["cora", "mnist", "cifar", "safenlp", "tin", "eransig", "erantanh"]


# --------------------------------------------------------------------------
# model loading + the constraint direction
# --------------------------------------------------------------------------
def load_models(bench: str):
    """{group_label: (inner_net, output_spec, per_sample_shape, safe_name)}."""
    from act.front_end.model_synthesis import synthesize_models_from_specs
    from act.front_end.verifiable_model import InputLayer
    from act.front_end.vnnlib_loader.create_specs import VNNLibSpecCreator

    cfg = BENCH[bench]
    specs = VNNLibSpecCreator().create_specs_for_data_model_pairs(
        categories=[cfg["category"]], max_instances=cfg["max_instances"],
    )
    if not specs:
        sys.exit(f"no VNNLIB specs for category {cfg['category']!r}")
    out = {}
    for model_id, wm in synthesize_models_from_specs(specs).items():
        raw = "_".join(map(str, model_id)) if isinstance(model_id, tuple) else str(model_id)
        raw = raw.replace("/", "_").replace("\\", "_")
        inp = next(m for m in wm.modules() if isinstance(m, InputLayer))
        net = wm.model.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        name = str(model_id[1]) if isinstance(model_id, tuple) and len(model_id) > 1 else raw
        mm = re.search(cfg["group_re"], name)
        label = mm.group(1) if mm else name
        # A regex that collapsed two groups to one label would silently drop a
        # group's counterexamples instead of failing.
        if label in out:
            sys.exit(f"[{bench}] group_re maps two model groups to {label!r}")
        out[label] = (net, wm.output_spec.spec,
                      tuple(inp.labeled_input.tensor.shape[1:]), raw)
    return out


def safe_name(raw: str, rep_dir: Path) -> str:
    """The directory name paper_cifar100_batch_ani.py would have written.

    Copied from that driver rather than imported: it is computed inline there,
    inside the group loop. Kept byte-identical including the truncation, which
    tinyimagenet's 120-character names actually hit.
    """
    budget = 259 - len(str(rep_dir.resolve())) - 26
    if len(raw) > budget:
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        raw = f"{raw[:max(8, budget - 9)]}_{digest}"
    return raw


def constraints(net, spec, shape, x_flat, inst, chunk=64):
    """Rows of `d severity / d x` -- each counterexample's affine constraint.

    Inside the open region where every ReLU sign and every argmax in the spec
    is fixed, severity is affine in x, so this gradient IS the constraint that
    counterexample imposes; two counterexamples from the same region return
    the same row. That is what makes the pairwise angle a redundancy measure
    and not just a summary of where the points are.

    `rows` pins every lane to instance `inst`: the spec carries one row per
    batched instance, and these counterexamples all belong to that one.
    """
    # Match the synthesized model's own dtype: ACT builds these under its
    # default dtype (float64 unless the run forced float32), and the dumps are
    # float16, so neither side can be assumed.
    dt = next(net.parameters()).dtype
    x = torch.as_tensor(np.asarray(x_flat, np.float64)).to(dt).reshape(len(x_flat), *shape)
    grads = []
    for i in range(0, len(x), chunk):
        xb = x[i:i + chunk].clone().requires_grad_(True)
        rows = torch.full((len(xb),), int(inst), dtype=torch.long)
        sev = spec.severity(net(xb), rows=rows)
        g, = torch.autograd.grad(sev.sum(), xb)
        # .cpu() before .numpy(): under a torch.device context the model and
        # everything built from it live on the accelerator, and numpy cannot
        # read that memory. A CPU-only run never reaches the conversion with a
        # device tensor, so this only shows up once a GPU is in play.
        grads.append(g.detach().reshape(len(xb), -1).to(torch.float64).cpu().numpy())
    return np.concatenate(grads)


def unit(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-30)


def exact_S(a):
    """Mean pairwise cosine distance over every pair -- no subsampling.

    Rarefaction was inherited from the Hill-number metric, whose estimator
    really is sample-size dependent; a mean pairwise distance is not, so
    subsampling only added variance. On cora it matched to three decimals.
    """
    C = unit(a) @ unit(a).T
    return float((1 - C[np.triu_indices(len(a), 1)]).mean())


# --------------------------------------------------------------------------
# per-benchmark analysis
# --------------------------------------------------------------------------
def analyse(bench: str, verbose=True):
    cfg = BENCH[bench]
    root = OUT / cfg["dir"]
    missing = [a for a in ARMS if not (root / a).is_dir()]
    if missing:
        sys.exit(f"[{bench}] no dump for arm(s) {missing} under {root}")

    models = load_models(bench)
    per, dumps, D = {}, {}, {}
    for arm in ARMS:
        arm_dir = root / arm
        D[arm] = sum(len(json.loads(p.read_text(encoding="utf-8"))
                         .get("distinct_instances_hit", []))
                     for p in sorted(arm_dir.glob("*/group_summary.json")))
        for grp, (net, spec, shape, raw) in models.items():
            npz = arm_dir / safe_name(raw, arm_dir) / "ce_sample.npz"
            if not npz.exists():
                continue
            d = np.load(npz)
            dumps[(arm, grp)] = d
            for k, inst in enumerate(d["instances"]):
                sel = d["instance"] == inst
                n = int(sel.sum())
                if n < 2:  # a single point spans no pair
                    continue
                a = constraints(net, spec, shape, d["x"][sel], int(inst))
                per.setdefault((grp, int(inst)), {})[arm] = (
                    exact_S(a), n, int(d["ce_total"][k]))

    keys = sorted(k for k in per if len(per[k]) == len(ARMS))
    # An empty intersection is a result, not a crash: on cifar100 the baseline
    # breaks 2 instances against the best arm's ~11, and the two need not be a
    # subset. Everything paired is then simply unavailable and only the
    # per-arm view below can be reported.
    means = {a: float(np.mean([per[k][a][0] for k in keys])) for a in ARMS} if keys else {}
    wins = {a: sum(1 for k in keys if max(ARMS, key=lambda b: per[k][b][0]) == a)
            for a in ARMS} if keys else {}
    # Paired (`means`, over `keys`) is the comparison; unpaired is the coverage.
    # On the benchmarks where the baseline breaks 4 instances and the best arm
    # breaks 30, the paired set is those same 4 and it is the only set on which
    # the arms can be differenced at all -- but it also discards most of what
    # the better arms found, so both are reported.
    own = {a: sorted(k for k in per if a in per[k]) for a in ARMS}
    means_all = {a: float(np.mean([per[k][a][0] for k in own[a]])) if own[a] else float("nan")
                 for a in ARMS}

    # S's sampling unit is the INSTANCE, not the trial: each instance's S is
    # already exact over every pair of its counterexamples, and the arm-level
    # number is an equal-weight mean over instances.  So the error bar that
    # belongs on S is the standard error ACROSS INSTANCES -- running more
    # trials would not shrink it, it would only enlarge the instance set.
    def _se(vals):
        v = np.asarray(vals, float)
        return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
    se = {a: _se([per[k][a][0] for k in keys]) for a in ARMS} if keys else {}
    se_all = {a: _se([per[k][a][0] for k in own[a]]) for a in ARMS}
    if verbose:
        print(f"\n[{bench}] {len(keys)} instances broken by all {len(ARMS)} arms (>=2 CEs each)")
        print(f"{'arm':<20}{'D':>6}{'CE rows':>10}{'pairs':>12}"
              f"{'S(paired)':>11}{'wins':>6}{'S(own)':>9}{'n_own':>7}")
        for a in ARMS:
            rows = sum(per[k][a][1] for k in keys)
            pairs = sum(per[k][a][1] * (per[k][a][1] - 1) // 2 for k in keys)
            sp = f"{means[a]:>11.3f}{wins[a]:>6}" if keys else f"{'--':>11}{'--':>6}"
            print(f"{a:<20}{D[a]:>6}{rows:>10,}{pairs:>12,}"
                  f"{sp}{means_all[a]:>9.3f}{len(own[a]):>7}")
    return dict(bench=bench, cfg=cfg, models=models, dumps=dumps, per=per,
                keys=keys, D=D, means=means, wins=wins,
                means_all=means_all, n_own={a: len(own[a]) for a in ARMS},
                se=se, se_all=se_all)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def figures(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bench, cfg, per, keys = R["bench"], R["cfg"], R["per"], R["keys"]
    D, means, wins, means_all = R["D"], R["means"], R["wins"], R["means_all"]
    se, se_all = R["se"], R["se_all"]

    # ---- example instance: equal point count, median-representative ---------
    # With an empty intersection there is no instance every arm can be drawn
    # on, so fall back to the most widely broken one and draw only the arms
    # that broke it -- stated in the caption, since a missing panel would
    # otherwise read as an arm that found nothing anywhere.
    # EXAMPLE_OVERRIDE lets a specific instance be requested.  The default
    # rule (median-representative) is deliberately effect-blind: picking the
    # instance with the widest gap would be choosing the example by the very
    # quantity the figure exists to show.  An override is therefore stated in
    # the caption, so a reader knows the choice was made by hand.
    forced = EXAMPLE_OVERRIDE.get(bench)
    pool = keys or sorted(per, key=lambda k: (-len(per[k]), k))[:1]
    panel_arms = [a for a in ARMS if a in per[pool[0]]] if not keys else ARMS
    cand = [k for k in pool if min(per[k][a][1] for a in panel_arms) >= 60] or pool
    med = {a: np.median([per[k][a][0] for k in pool if a in per[k]]) for a in panel_arms}
    GROUP, INST = min(cand, key=lambda k: sum(abs(per[k][a][0] - med[a]) for a in panel_arms))
    if forced:
        want = (forced[0], int(forced[1]))
        if want in per and all(a in per[want] for a in panel_arms):
            GROUP, INST = want
        else:
            print(f"  [warn] example override {forced} is not a paired instance "
                  f"here; keeping {GROUP} #{INST}")
            forced = None
    NP = min(per[(GROUP, INST)][a][1] for a in panel_arms)
    sub = [a for a in panel_arms if per[(GROUP, INST)][a][1] > NP]
    print(f"\nexample instance = {GROUP} #{INST}, {NP} points per panel"
          + (f" (subsampled from {[per[(GROUP, INST)][a][1] for a in sub]} in "
             f"{', '.join(SHORT[a] for a in sub)})" if sub else " -- no subsampling"))
    if len(panel_arms) < len(ARMS):
        print(f"  arms without >=2 CEs on this instance: "
              f"{[SHORT[a] for a in ARMS if a not in panel_arms]}")

    net, spec, shape, _ = R["models"][GROUP]
    rng = np.random.default_rng(0)
    dirs = {}
    for arm in panel_arms:
        d = R["dumps"][(arm, GROUP)]
        rows = d["x"][d["instance"] == INST]
        # Panels draw an EQUAL number of points: a panel with 250 dots beside
        # one with 66 reads as more spread out for a reason that has nothing
        # to do with spread. S in the titles is still exact over the full set.
        if len(rows) > NP:
            rows = rows[rng.choice(len(rows), NP, replace=False)]
        dirs[arm] = unit(constraints(net, spec, shape, rows, INST))

    stack = np.vstack([dirs[a] for a in panel_arms])
    mu = stack.mean(0)
    _, sv, Vt = np.linalg.svd(stack - mu, full_matrices=False)
    ev = sv ** 2 / (sv ** 2).sum()
    proj = {a: (dirs[a] - mu) @ Vt[:2].T for a in panel_arms}
    allp = np.vstack(list(proj.values()))
    pad = 0.08 * (allp.max(0) - allp.min(0))
    lo, hi = allp.min(0) - pad, allp.max(0) + pad

    fig, axes = plt.subplots(1, len(panel_arms), figsize=(max(4.1 * len(panel_arms), 11.0), 4.6),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    for ax, arm in zip(axes, panel_arms):
        p = proj[arm]
        ax.scatter(p[:, 0], p[:, 1], s=26 if NP > 120 else 30,
                   alpha=.7 if NP > 120 else .8, color=COL[arm],
                   edgecolor="white", linewidth=.5)
        ax.set_title(f"{TITLES[arm]}\nS = {per[(GROUP, INST)][arm][0]:.3f}",
                     fontsize=10.5, color=INK)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}%)", fontsize=9.5, color=INK2)
        ax.axhline(0, lw=.5, c=GRID)
        ax.axvline(0, lw=.5, c=GRID)
        ax.tick_params(labelsize=8, colors=INK2)
        for s in ax.spines.values():
            s.set_color(GRID)
    axes[0].set_ylabel(f"PC2 ({ev[1] * 100:.1f}%)", fontsize=9.5, color=INK2)
    fig.suptitle(f"Counterexample constraint directions -- {cfg['category']} "
                 f"{GROUP} instance {INST} (n={NP} per panel, shared PCA basis)",
                 fontsize=12, color=INK)
    if forced:
        caption = (f"Example instance CHOSEN BY HAND out of the {len(keys)} "
                   f"instances all {len(ARMS)} arms broke -- NOT the default "
                   "median-representative pick.")
    else:
        caption = ("Example instance, picked as the median-representative one of "
                   f"the {len(keys)} instances all {len(ARMS)} arms broke."
                   if keys else
                   f"No instance was broken by all {len(ARMS)} arms, so this is the "
                   "most widely broken one; arms absent from it are not drawn.")
    import textwrap
    full = (caption + " S in each title is exact over that arm's full "
            "counterexample set, not over the points drawn.")
    wrapped = textwrap.fill(full, width=int(fig.get_figwidth() * 15))
    fig.text(.5, .012, wrapped, ha="center", va="bottom",
             fontsize=8.2, color=MUTED, linespacing=1.45)
    fig.tight_layout(rect=[0, .035 + .028 * wrapped.count(chr(10)), 1, 1])
    f1 = OUT / f"fig_ce_diversity_{TAG}_{bench}.png"
    fig.savefig(f1, dpi=170, facecolor="white")
    plt.close(fig)

    # ---- companion: every instance, and the (D, S) plane --------------------
    fig2, (axA, axB) = plt.subplots(1, 2, figsize=(12.2, 4.7),
                                    gridspec_kw=dict(width_ratios=[1.3, 1]))
    xs = np.arange(len(ARMS))
    if keys:
        for k in keys:
            axA.plot(xs, [per[k][a][0] for a in ARMS], color=MUTED, lw=.8, alpha=.4,
                     marker="o", ms=2.8, mfc="white", mec=MUTED, mew=.6, zorder=1)
        line = [means[a] for a in ARMS]
        n_each = " / ".join(str(R["n_own"][a]) for a in ARMS)
        title = (f"Intra-instance diversity on the PAIRED set: {len(keys)} instance(s)\n"
                 f"every arm broke  (each arm alone broke {n_each})\n"
                 f"thin line = one instance; {SHORT[ARMS[-1]]} widest in "
                 f"{wins[ARMS[-1]]} of {len(keys)}")
    else:
        # Unpaired: each arm over the instances IT broke. The connecting line
        # is drawn thinner and the title says so, because joining means over
        # different instance sets is not the paired comparison above it.
        rng2 = np.random.default_rng(1)
        for i, a in enumerate(ARMS):
            v = [per[k][a][0] for k in per if a in per[k]]
            axA.scatter(i + rng2.uniform(-.09, .09, len(v)), v, s=16, color=MUTED,
                        alpha=.55, edgecolor="none", zorder=1)
        line = [means_all[a] for a in ARMS]
        title = ("Intra-instance diversity, each arm over the instances IT broke\n"
                 f"no instance was broken by all {len(ARMS)} arms -- NOT a paired comparison")
    axA.plot(xs, line, color=INK, lw=2.2 if keys else 1.2,
             ls="-" if keys else "--", zorder=3)
    for i, a in enumerate(ARMS):
        axA.plot(xs[i], line[i], marker="o", ms=10, color=COL[a],
                 mec="white", mew=1.6, zorder=4)
        axA.annotate(f"{line[i]:.3f}", (xs[i], line[i]), textcoords="offset points",
                     xytext=(0, 13), ha="center", fontsize=10, color=INK, weight="bold")
    axA.set_xticks(xs)
    axA.set_xticklabels([SHORT[a] for a in ARMS], fontsize=8.6, color=INK2)
    axA.set_xlim(-.35, len(ARMS) - .65)
    axA.set_ylabel("S  (mean pairwise cosine distance)", fontsize=10, color=INK2)
    axA.set_title(title, fontsize=10.5, color=INK)
    axA.grid(axis="y", color=GRID, lw=.6)
    axA.set_axisbelow(True)

    sy = means if keys else means_all
    sy_err = se if keys else se_all
    # x is the campaign mean where one exists: the single diversity run cannot
    # rank arms, and an integer count on this axis silently implied it could.
    dv, dsd, dsrc = campaign_or_run(bench, D)
    for a in ARMS:
        axB.errorbar(dv[a], sy[a],
                     xerr=dsd[a] or None,
                     yerr=sy_err[a] if np.isfinite(sy_err.get(a, float("nan")))
                          else None,
                     fmt="o", ms=13, color=COL[a], mec="white", mew=1.8,
                     ecolor=MUTED, elinewidth=1.1, capsize=4, zorder=3)
        axB.annotate(SHORT[a], (dv[a], sy[a]), textcoords="offset points",
                     xytext=(0, 19), ha="center", fontsize=9.5, color=INK)
    axB.set_xlabel("D  -- instances broken (inter-instance), %s%s"
                   % (dsrc, " +- sd" if dsd[ARMS[0]] else " -- no campaign"),
                   fontsize=10, color=INK2)
    axB.set_ylabel("S  -- mean cosine distance (intra-instance)", fontsize=10, color=INK2)
    axB.set_title("The two axes are not combined into one score:\n"
                  "different units, and they can rank the arms differently",
                  fontsize=10.5, color=INK)
    axB.grid(color=GRID, lw=.6)
    axB.set_axisbelow(True)
    span = max(dv.values()) - min(dv.values())
    dx = span * .16 + max(dsd.values()) + (1 if span else .5)
    axB.set_xlim(min(dv.values()) - dx, max(dv.values()) + dx)
    # Proportional pad: S runs ~0.3-0.6 on cora/mnist but ~0.02 on safenlp,
    # where a fixed pad would open the axis below zero and make a real spread
    # look like noise around nothing. S is a distance, so never go below 0.
    mv = list(sy.values())
    dy = max(.25 * (max(mv) - min(mv)), .04 * max(mv), 1e-3)
    axB.set_ylim(max(0.0, min(mv) - dy), max(mv) + dy)

    for ax in (axA, axB):
        ax.tick_params(labelsize=8.5, colors=INK2)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)

    fig2.suptitle(f"{cfg['label']}  --  S exact over every counterexample, "
                  "all pairs, no rarefaction", fontsize=10.5, color=INK)
    fig2.tight_layout(rect=[0, 0, 1, .93])
    f2 = OUT / f"fig_ce_diversity_{TAG}_{bench}_S.png"
    fig2.savefig(f2, dpi=170, facecolor="white")
    plt.close(fig2)

    json.dump({"bench": bench, "arms": ARMS, "D": D,
               "mean_S": means, "se_S": R["se"],
               "mean_S_own_instances": R["means_all"], "se_S_own": R["se_all"],
               "n_own_instances": R["n_own"], "wins": wins,
               "n_common_instances": len(keys),
               "example": [GROUP, INST, NP],
               "per_instance": {f"{g}|{i}": per[(g, i)] for g, i in keys}},
              open(OUT / f"ce_diversity_{TAG}_{bench}.json", "w"), indent=1)
    print(f"explained variance PC1={ev[0] * 100:.1f}%  PC2={ev[1] * 100:.1f}%  "
          f"PC1+2={ev[:2].sum() * 100:.1f}%")
    print("saved", f1)
    print("saved", f2)


def campaign_or_run(bench, D_run):
    """D for plotting: the campaign mean where one exists, else this run.

    A single run cannot rank arms -- on mnist the campaign has statehpgd
    6.83 over statebase 5.53, yet one trial reverses that about a quarter of
    the time.  Returns (value, sd, label); sd is 0 for the n=1 fallback, and
    the label says which it was so no figure can present one as the other.
    """
    camp = CAMPAIGN_D.get(bench)
    if camp and all(a in camp for a in ARMS):
        return ({a: camp[a][0] for a in ARMS},
                {a: camp[a][1] for a in ARMS},
                "n=%d" % camp[ARMS[0]][2])
    return ({a: float(D_run[a]) for a in ARMS},
            {a: 0.0 for a in ARMS}, "n=1")


def n_str(d):
    return "/".join(str(d[a]) for a in ARMS)


def combined(benches):
    """One figure over every benchmark: the (D, S) plane, normalised.

    Both axes are shown relative to that benchmark's own best arm, because D is
    a raw count and the benchmarks have different instance budgets -- plotting
    the counts together would compare 34-of-200 against 16-of-90.

    The two axes come from DIFFERENT sources, on purpose:

    * D is the campaign mean (CAMPAIGN_D), not the single diversity run.  One
      run is far too noisy to rank arms: on mnist the campaign has statehpgd
      6.83 over statebase 5.53, but a single trial reverses that about one time
      in four, and the diversity run is one such trial (5 vs 6).
    * S cannot come from a campaign, because it is computed from the dumped
      counterexamples of one run.  It does not need to: S's sampling unit is
      the INSTANCE, and each instance's S is already exact over all of its
      pairs.  The error bar that belongs on S is therefore the SE across
      instances, which more trials would not shrink.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    have = [b for b in ORDER if b in benches]
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.2))
    MARK = dict(cora="o", mnist="s", cifar="^", safenlp="D")

    # ---- left: D, campaign mean where one exists ------------------------
    # same resolver the per-benchmark (D, S) panel uses, so the two figures
    # can never disagree about what D means
    d_src, d_val, d_err = {}, {}, {}
    for b in have:
        d_val[b], d_err[b], d_src[b] = campaign_or_run(b, benches[b]["D"])

    # ---- right: S over EVERY instance the arm broke, +- SE across them ---
    s_val = {b: benches[b]["means_all"] for b in have}
    s_err = {b: benches[b]["se_all"] for b in have}
    s_n = {b: benches[b]["n_own"] for b in have}

    # An arm-level S resting on one or two instances is a point, not a mean.
    MIN_INST = 3
    thin = {b: min(s_n[b].values()) < MIN_INST for b in have}

    # paired S: the only set on which the arms can actually be differenced
    p_val = {b: benches[b]["means"] for b in have}
    p_err = {b: benches[b]["se"] for b in have}
    p_n = {b: len(benches[b]["keys"]) for b in have}
    p_thin = {b: p_n[b] < MIN_INST for b in have}

    for ax, key in zip(axes, ("D", "Sp", "S")):
        val = {"D": d_val, "Sp": p_val, "S": s_val}[key]
        err = {"D": d_err, "Sp": p_err, "S": s_err}[key]
        for b in have:
            v, e = val[b], err[b]
            if not v:          # empty all-arm intersection
                continue
            top = max(v.values())
            if not top:
                continue
            ys = [v[a] / top for a in ARMS]
            es = [(e[a] / top if np.isfinite(e[a]) else 0.0) for a in ARMS]
            dim = (key == "S" and thin[b]) or (key == "Sp" and p_thin[b])
            ax.errorbar(np.arange(len(ARMS)), ys, yerr=es, color=MUTED,
                        lw=1.0, alpha=.28 if dim else .75,
                        ls=":" if dim else "-", capsize=3, zorder=1)
            for i, a in enumerate(ARMS):
                ax.scatter(i, ys[i], s=85, color=COL[a], marker=MARK[b],
                           edgecolor="white", linewidth=1.2,
                           alpha=.32 if dim else 1.0, zorder=3)
        ax.set_xticks(np.arange(len(ARMS)))
        ax.set_xticklabels([SHORT[a] for a in ARMS], fontsize=8.6, color=INK2)
        ax.set_xlim(-.4, len(ARMS) - .6)
        ax.grid(axis="y", color=GRID, lw=.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8.5, colors=INK2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)

    axes[0].set_ylabel("D  -- instances broken, campaign mean +- sd\n"
                       "(relative to best arm)", fontsize=9.5, color=INK2)
    axes[1].set_ylabel("S  -- PAIRED: only the instances every arm broke,\n"
                       "+- SE across instances (relative to best arm)",
                       fontsize=9.5, color=INK2)
    axes[2].set_ylabel("S  -- UNPAIRED: every instance that arm broke,\n"
                       "+- SE across instances (relative to best arm)",
                       fontsize=9.5, color=INK2)
    for ax, t in zip(axes, PANEL_TITLES[ARM_SET_NAME]):
        ax.set_title(t, fontsize=10.5, color=INK)

    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([], [], color=MUTED, marker=MARK[b], ls="none",
                                   ms=7, label=b + " (" + d_src[b] + ")")
                            for b in have],
                   fontsize=8.2, frameon=False, loc="lower right", labelcolor=INK2)
    axes[1].legend(handles=[Line2D([], [], color=MUTED, marker=MARK[b], ls="none",
                                   ms=7, alpha=.35 if p_thin[b] else 1.0,
                                   label=b + " (n=" + str(p_n[b])
                                         + (", too few" if p_thin[b] else "") + ")")
                            for b in have],
                   fontsize=8.2, frameon=False, loc="lower right", labelcolor=INK2)
    axes[2].legend(handles=[Line2D([], [], color=MUTED, marker=MARK[b], ls="none",
                                   ms=7, alpha=.35 if thin[b] else 1.0,
                                   label=b + " (n=" + n_str(s_n[b])
                                         + (", too few" if thin[b] else "") + ")")
                            for b in have],
                   fontsize=8.2, frameon=False, loc="lower right", labelcolor=INK2)

    fig.suptitle("%d-arm counterexample diversity across " % len(ARMS)
                 + ", ".join(BENCH[b]["category"] for b in have)
                 + "  --  each benchmark scaled to its own best arm",
                 fontsize=10.5, color=INK)
    notes = []
    no_camp = [b for b in have if d_src[b] == "n=1"]
    if no_camp:
        notes.append(", ".join(no_camp) + ": no distinct campaign exists -- D is one "
                     "run, no error bar.")
    star = [b for b in have if thin[b]]
    if star:
        notes.append(", ".join(star) + ": S faded -- an arm rests on under %d "
                     "instances, which is a point, not a mean." % MIN_INST)
    notes.append("Panels 2 and 3 are the same quantity over different instance sets "
                 "and can rank the arms differently: paired is the only set the arms "
                 "can be differenced on, unpaired uses all the data.")
    # one note per line -- a single joined line overflows the figure width
    fig.text(.5, .012, chr(10).join(notes), ha="center", va="bottom",
             fontsize=8.0, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=[0, .04 + .035 * len(notes), 1, .93])
    f = OUT / ("fig_ce_diversity_%s_all.png" % TAG)
    fig.savefig(f, dpi=170, facecolor="white")
    plt.close(fig)
    print("saved", f)

    rows = {b: dict(D_run=benches[b]["D"], D_campaign=d_val[b], D_source=d_src[b],
                    S_paired=benches[b]["means"], S_paired_se=benches[b]["se"],
                    S_all=s_val[b], S_all_se=s_err[b], n_own=s_n[b],
                    n_common=len(benches[b]["keys"]),
                    S_too_few_instances=bool(thin[b]),
                    wins=benches[b]["wins"]) for b in have}
    json.dump({"arms": ARMS, "benchmarks": rows},
              open(OUT / ("ce_diversity_%s_all.json" % TAG), "w"), indent=1)

    head = "".join("%22s" % SHORT[a] for a in ARMS)
    print("\n" + " " * 10 + head)
    for b in have:
        cells = ""
        for a in ARMS:
            se = s_err[b][a]
            se_s = ("+-%.3f" % se) if np.isfinite(se) else "    --"
            cells += "%8.2f%8.3f%6s" % (d_val[b][a], s_val[b][a], se_s)
        print("%-10s%s" % (b, cells))
    print(" " * 10 + "".join("%8s%8s%6s" % ("D", "S", "SE") for _ in ARMS))
    print("\nD source: " + ", ".join(b + " " + d_src[b] for b in have))
    print("S source: one diversity run; n = instances that arm broke -- "
          + ", ".join(b + " " + n_str(s_n[b]) for b in have))


def main():
    global ARMS, TAG, ARM_SET_NAME
    argv = [a for a in sys.argv[1:]]
    # --arms {energy,portfolio}: which arm set to read.  Defaults to the
    # original four so every existing invocation keeps its behaviour and its
    # output filenames.
    if "--arms" in argv:
        i = argv.index("--arms")
        if i + 1 >= len(argv):
            sys.exit("--arms needs a value; expected one of " + str(list(ARM_SETS)))
        name = argv[i + 1]
        if name not in ARM_SETS:
            sys.exit(f"unknown arm set {name!r}; expected one of {list(ARM_SETS)}")
        ARMS = ARM_SETS[name]
        ARM_SET_NAME = name
        TAG = f"{len(ARMS)}arm" if name == "energy" else f"{len(ARMS)}arm_{name}"
        del argv[i:i + 2]
    # --example <bench>:<group>#<inst>  (repeatable) pins the PCA example
    while "--example" in argv:
        k = argv.index("--example")
        if k + 1 >= len(argv):
            sys.exit("--example needs <bench>:<group>#<instance>")
        spec = argv[k + 1]
        try:
            bench_, rest = spec.split(":", 1)
            grp, inst = rest.rsplit("#", 1)
            EXAMPLE_OVERRIDE[bench_] = (grp, int(inst))
        except ValueError:
            sys.exit(f"bad --example {spec!r}; want <bench>:<group>#<instance>")
        del argv[k:k + 2]
    which = argv[0] if argv else "all"
    todo = ORDER if which == "all" else [which]
    for b in todo:
        if b not in BENCH:
            sys.exit(f"unknown benchmark {b!r}; expected one of {list(BENCH)} or 'all'")
    results = {}
    for b in todo:
        results[b] = analyse(b)
        figures(results[b])
    if len(results) > 1:
        combined(results)


if __name__ == "__main__":
    main()
