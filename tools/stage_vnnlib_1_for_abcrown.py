"""Stage a VNNLIB-2.0-format category into a 1.0-format copy the officially-
released alpha-beta-CROWN verifier can actually parse (confirmed: its own
read_vnnlib.py hard-fails on `(vnnlib-version <2.0>)`). Copies onnx/ and
instances.csv as-is, converts every vnnlib/*.vnnlib file via
tools/convert_vnnlib_2_to_1.py's convert().

Usage:
    python tools/stage_vnnlib_1_for_abcrown.py <category> <dest_root>
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from tools.convert_vnnlib_2_to_1 import convert  # noqa: E402

DATA_ROOT = WORKSPACE_ROOT / "data" / "vnnlib"


def main() -> None:
    category, dest_root_str = sys.argv[1], sys.argv[2]
    src_root = DATA_ROOT / category
    dest_root = Path(dest_root_str) / category

    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True)

    shutil.copytree(src_root / "onnx", dest_root / "onnx")
    shutil.copy2(src_root / "instances.csv", dest_root / "instances.csv")

    vnnlib_src_dir = src_root / "vnnlib"
    vnnlib_dst_dir = dest_root / "vnnlib"
    n_ok, n_err = 0, 0
    for src_file in vnnlib_src_dir.rglob("*.vnnlib"):
        rel = src_file.relative_to(vnnlib_src_dir)
        dst_file = vnnlib_dst_dir / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            converted = convert(src_file.read_text(encoding="utf-8"))
            dst_file.write_text(converted, encoding="utf-8")
            n_ok += 1
        except Exception as exc:
            print(f"  ERROR converting {rel}: {exc!r}")
            n_err += 1

    print(f"{category}: converted {n_ok} vnnlib file(s), {n_err} error(s) -> {dest_root}")


if __name__ == "__main__":
    main()
