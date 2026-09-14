"""Convert a VNNLIB 1.0 flat benchmark into the 2.0 form ACT reads.

ACT is VNNLIB 2.0-only, so 1.0 categories (mnist_fc among them) fail at load
with "VNNLIB 1.0 flat format is no longer supported". The two formats differ
mechanically, not semantically:

    1.0                               2.0
    (declare-const X_0 Real) x D      (vnnlib-version <2.0>)
    (declare-const Y_0 Real) x K      (declare-network N
                                          (declare-input  X float32 [1, D])
                                          (declare-output Y float32 [1, K]))
    (assert (>= X_0 v))               (assert (>= X[0,0] v))
    (assert (>= Y_5 Y_6))             (assert (>= Y[0,5] Y[0,6]))

So: count the declarations to recover D and K, emit the header, drop the
declare-consts, and reindex every variable reference. Nothing about the
constraints themselves changes.

Writes a NEW category directory rather than editing in place, so the original
benchmark stays byte-identical and a mistake here cannot corrupt it.

    python tools/convert_vnnlib_1_to_2.py mnist_fc mnist_fc_v2
"""

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "vnnlib"

DECL = re.compile(r"^\s*\(declare-const\s+([XY])_(\d+)\s+Real\s*\)\s*$")
# Word-boundary on both sides: X_1 must not match inside X_10.
REF = re.compile(r"\b([XY])_(\d+)\b")


def convert_text(text: str, input_shape=None) -> str:
    d = k = 0
    body = []
    for line in text.splitlines():
        m = DECL.match(line)
        if m:
            idx = int(m.group(2)) + 1
            if m.group(1) == "X":
                d = max(d, idx)
            else:
                k = max(k, idx)
            continue
        body.append(line)
    if d == 0 or k == 0:
        raise ValueError(f"no X_/Y_ declarations found (D={d}, K={k})")

    shape = list(input_shape) if input_shape else [1, d]
    header = (
        "(vnnlib-version <2.0>)\n\n"
        "(declare-network N\n"
        f"    (declare-input  X float32 [{', '.join(str(s) for s in shape)}])\n"
        f"    (declare-output Y float32 [1, {k}])\n"
        ")\n"
    )
    # X_7 -> X[0,7]; the leading 0 is the batch index every 2.0 file carries.
    converted = REF.sub(lambda m: f"{m.group(1)}[0,{m.group(2)}]", "\n".join(body))
    return header + converted.lstrip("\n") + "\n"


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    src, dst = ROOT / sys.argv[1], ROOT / sys.argv[2]
    if not src.is_dir():
        raise SystemExit(f"missing category: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    for name in ("onnx", "instances.csv", "info.json"):
        s = src / name
        if not s.exists():
            continue
        t = dst / name
        if t.exists():
            continue
        (shutil.copytree if s.is_dir() else shutil.copy2)(s, t)

    ok = bad = 0
    for f in sorted((src / "vnnlib").rglob("*.vnnlib")):
        out = dst / "vnnlib" / f.relative_to(src / "vnnlib")
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            out.write_text(convert_text(f.read_text(encoding="utf-8")), encoding="utf-8")
            ok += 1
        except Exception as exc:
            bad += 1
            print(f"  FAILED {f.name}: {exc}")
    print(f"converted {ok} file(s), {bad} failed -> {dst}")


if __name__ == "__main__":
    main()
