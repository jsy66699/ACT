"""Fold the ERAN nets' embedded MNIST normalization into the first Gemm.

The downloaded ONNX graphs start with `Sub(mean) -> Div(std) -> Flatten -> Gemm`.
ACT's torch2act cannot build that leading broadcast ("Var-var 'sub' size
mismatch (784 vs 1)"), so this writes an equivalent copy of each model with
the normalization folded into the first linear layer:

    W ((x - m) / s) + b  ==  (W / s) x + (b - W m / s)

Inputs stay raw [0, 1] pixels, so the generated VNNLIB specs are unchanged.
Outputs land in onnx_folded/ with instances_folded.csv; the originals are kept
untouched.  Equivalence is checked against the original graph on the first 100
MNIST test images.

Run from the repo root:
    python data/vnnlib/eran_sigmoid_tanh_mlp/fold_normalization.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

HERE = Path(__file__).resolve().parent
SRC = HERE / "onnx"
DST = HERE / "onnx_folded"


def const_value(graph, name):
    for node in graph.node:
        if node.op_type == "Constant" and node.output[0] == name:
            return numpy_helper.to_array(node.attribute[0].t)
    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    return None


def fold(path: Path) -> Path:
    model = onnx.load(path)
    g = model.graph

    sub = next(n for n in g.node if n.op_type == "Sub")
    div = next(n for n in g.node if n.op_type == "Div")
    gemm = next(n for n in g.node if n.op_type == "Gemm")
    mean = const_value(g, sub.input[1]).reshape(-1)
    std = const_value(g, div.input[1]).reshape(-1)
    assert mean.size == 1 and std.size == 1, (mean.shape, std.shape)
    m, s = float(mean[0]), float(std[0])

    inits = {i.name: i for i in g.initializer}
    W = numpy_helper.to_array(inits[gemm.input[1]]).astype(np.float64)
    b = numpy_helper.to_array(inits[gemm.input[2]]).astype(np.float64)
    trans_b = any(a.name == "transB" and a.i == 1 for a in gemm.attribute)
    # rows of the matrix that multiply the input, whichever layout Gemm uses
    W_in = W if trans_b else W.T          # [out, in]
    W_new = W_in / s
    b_new = b - W_in @ (np.full(W_in.shape[1], m) / s)
    W_out = W_new if trans_b else W_new.T

    inits[gemm.input[1]].CopyFrom(
        numpy_helper.from_array(W_out.astype(np.float32), gemm.input[1])
    )
    inits[gemm.input[2]].CopyFrom(
        numpy_helper.from_array(b_new.astype(np.float32), gemm.input[2])
    )

    # drop Sub/Div (and their Constant producers) and rewire Flatten to the input
    flatten = next(n for n in g.node if n.op_type == "Flatten")
    flatten.input[0] = g.input[0].name
    drop = {sub.name, div.name}
    dead = {sub.input[1], div.input[1]}
    keep = [
        n for n in g.node
        if n.name not in drop and not (n.op_type == "Constant" and n.output[0] in dead)
    ]
    del g.node[:]
    g.node.extend(keep)

    onnx.checker.check_model(model)
    out = DST / path.name
    onnx.save(model, out)
    return out


def check(orig: Path, folded: Path, images: np.ndarray) -> float:
    a = ort.InferenceSession(str(orig), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(folded), providers=["CPUExecutionProvider"])
    na, nb = a.get_inputs()[0].name, b.get_inputs()[0].name
    worst = 0.0
    for img in images:
        x = img.reshape(1, 1, 28, 28)
        ya = a.run(None, {na: x})[0]
        yb = b.run(None, {nb: x})[0]
        worst = max(worst, float(np.abs(ya - yb).max()))
    return worst


def main() -> None:
    DST.mkdir(exist_ok=True)
    csv_path = HERE.parents[1] / "csv" / "mnist_first_100_samples.csv"
    with csv_path.open() as f:
        rows = list(csv.reader(f))[1:]
    images = np.array([[float(v) for v in r[1:]] for r in rows], dtype=np.float32)

    for path in sorted(SRC.glob("*.onnx")):
        out = fold(path)
        print(f"{path.name}: max |orig - folded| over 100 images = {check(path, out, images):.3e}")

    src_rows = list(csv.reader((HERE / "instances.csv").open()))
    with (HERE / "instances_folded.csv").open("w", newline="") as f:
        csv.writer(f).writerows(
            [(r[0].replace("onnx/", "onnx_folded/"), r[1], r[2]) for r in src_rows]
        )
    print(f"wrote instances_folded.csv ({len(src_rows)} instances)")


if __name__ == "__main__":
    main()
