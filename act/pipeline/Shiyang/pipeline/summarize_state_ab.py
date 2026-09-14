"""Summarise a distinct/violations A/B written by paper_cifar100_batch_ani.

Reads every `<arm>/rep*/<group>/group_summary.json` under one results dir and
prints per-arm mean +- sd of violations and of distinct instances hit.

    python -m act.pipeline.Shiyang.pipeline.summarize_state_ab _smooth_state_ab
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4] / "act/pipeline/Shiyang/results"


def main() -> None:
    base = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "_smooth_state_ab")
    arms = sorted(p for p in base.iterdir() if p.is_dir())
    print(f"{'arm':<16} {'n':>2} {'violations':>22} {'distinct':>16} {'iters':>9}")
    for arm in arms:
        viol, dist, iters, hit = [], [], [], set()
        for f in sorted(arm.glob("rep*/*/group_summary.json")):
            d = json.loads(f.read_text())
            viol.append(d.get("violations", 0))
            hits = d.get("distinct_instances_hit", [])
            dist.append(len(hits))
            hit.update(hits)
            iters.append(d.get("iterations", 0))
        if not viol:
            continue

        def ms(v):
            return (f"{statistics.mean(v):>9.1f} +- {statistics.pstdev(v):<6.1f}"
                    if len(v) > 1 else f"{v[0]:>9.1f}{'':<10}")

        print(f"{arm.name:<16} {len(viol):>2} {ms(viol):>22} {ms(dist):>16} "
              f"{statistics.mean(iters):>9.0f}   union={sorted(hit)}")


if __name__ == "__main__":
    main()
