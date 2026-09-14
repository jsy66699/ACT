"""Batch-download a list of VNN-COMP 2026 VNNLIB benchmark categories (2.0
format) into data/vnnlib/<category>/, using ACT's own downloader
(act.front_end.vnnlib_loader.data_model_loader.download_vnnlib_category).

That downloader already does the "decompress + name each file per
instances.csv" step itself -- for every onnx/vnnlib path listed in the
category's instances.csv, it tries "<path>.gz" first (decompressing to
"<path>"), then "<path>" uncompressed, so files land on disk under exactly
the relative paths instances.csv references. No separate rename step needed.

Usage:
    python tools/batch_download_vnnlib_categories.py --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

DEFAULT_CATEGORIES = [
    "acasxu_2023",
    "cersyve",
    "challenging_certified_training_2026",
    "cifar100_2024",
    "cora_2024",
    "dist_shift_2023",
    "malbeware",
    "relusplitter_2026",
    "safenlp_2024",
    "sat_relu",
    "tinyimagenet_2024",
    "tllverifybench_2023",
    "vggnet16_2022",
    "vit_2023",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--force", action="store_true", help="Redownload even if already present locally.")
    args = parser.parse_args()

    from act.front_end.vnnlib_loader.data_model_loader import download_vnnlib_category

    results = []
    started = time.time()
    for i, category in enumerate(args.categories, 1):
        t0 = time.time()
        print(f"\n[{i}/{len(args.categories)}] === {category} ===", flush=True)
        try:
            result = download_vnnlib_category(category, force_redownload=args.force)
        except Exception as exc:
            result = {"status": "error", "message": repr(exc)}
        elapsed = time.time() - t0
        result["category"] = category
        result["elapsed_s"] = round(elapsed, 1)
        results.append(result)
        print(
            f"[{i}/{len(args.categories)}] {category}: status={result.get('status')} "
            f"instances={result.get('num_instances')} elapsed={elapsed:.1f}s "
            f"message={result.get('message', '')}",
            flush=True,
        )

    total_elapsed = time.time() - started
    print(f"\n=== Summary ({total_elapsed:.1f}s total) ===")
    for r in results:
        print(f"  {r['category']:45s} {r.get('status'):8s} instances={r.get('num_instances')}")


if __name__ == "__main__":
    main()
