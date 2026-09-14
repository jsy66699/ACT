"""Generate VNNLIB specs + instances.csv for the ERAN Sigmoid/Tanh MNIST MLPs.

Same recipe as the GenBaB / alpha-beta-CROWN configs these models ship with:
the first 100 MNIST test images, L-inf epsilon per model, and only the images
the model classifies correctly.  Specs are written in VNNLIB 2.0 (the only
form ACT reads); use tools/convert_vnnlib_2_to_1.py for 1.0-only verifiers.
The models embed MNIST normalization
(Sub/Div) in the ONNX graph, so the specs are written on raw [0, 1] pixels,
exactly like the mnist_fc benchmark.

Run from the repo root:
    python data/vnnlib/eran_sigmoid_tanh_mlp/generate_specs.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
MNIST_CSV = HERE.parents[1] / "csv" / "mnist_first_100_samples.csv"
TIMEOUT = 300

# epsilon values come from the GenBaB configs (huggingface.co/datasets/zhouxingshi/GenBaB)
MODELS = [
    ("ffnnSIGMOID__Point_6x100.onnx", 0.015),
    ("ffnnSIGMOID__Point_6x200.onnx", 0.012),
    ("ffnnSIGMOID__Point_9x100.onnx", 0.015),
    ("ffnnTANH__Point_6x100.onnx", 0.006),
    ("ffnnTANH__Point_6x200.onnx", 0.002),
    ("ffnnTANH__Point_9x100.onnx", 0.006),
]


def load_mnist() -> tuple[np.ndarray, np.ndarray]:
    with MNIST_CSV.open() as f:
        rows = list(csv.reader(f))[1:]
    labels = np.array([int(r[0]) for r in rows], dtype=np.int64)
    images = np.array([[float(v) for v in r[1:]] for r in rows], dtype=np.float32)
    return images, labels


def write_spec(path: Path, image: np.ndarray, label: int, eps: float) -> None:
    lo = np.clip(image - eps, 0.0, 1.0)
    hi = np.clip(image + eps, 0.0, 1.0)
    lines = [
        "(vnnlib-version <2.0>)",
        "",
        "(declare-network N",
        f"    (declare-input  X float32 [1, {image.size}])",
        "    (declare-output Y float32 [1, 10])",
        ")",
        f"; MNIST property with label: {label}.",
        "",
        "; Input constraints:",
    ]
    for i in range(image.size):
        lines.append(f"(assert (<= X[0,{i}] {hi[i]:.9f}))")
        lines.append(f"(assert (>= X[0,{i}] {lo[i]:.9f}))")
        lines.append("")
    lines += ["", "; Output constraints:", "(assert (or"]
    for j in range(10):
        if j != label:
            lines.append(f"    (and (>= Y[0,{j}] Y[0,{label}]))")
    lines.append("))")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    images, labels = load_mnist()
    instances = []
    for model_name, eps in MODELS:
        model_path = HERE / "onnx" / model_name
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        stem = model_path.stem
        spec_dir = HERE / "vnnlib" / stem
        spec_dir.mkdir(parents=True, exist_ok=True)

        n_correct = 0
        for idx, (image, label) in enumerate(zip(images, labels)):
            x = image.reshape(1, 1, 28, 28)
            pred = int(np.argmax(sess.run(None, {in_name: x})[0]))
            if pred != label:
                continue  # misclassified clean image: no robustness property to state
            n_correct += 1
            spec_name = f"prop_{idx}_{eps}.vnnlib"
            write_spec(spec_dir / spec_name, image, int(label), eps)
            instances.append(
                (f"onnx/{model_name}", f"vnnlib/{stem}/{spec_name}", TIMEOUT)
            )
        print(f"{model_name}: eps={eps}, {n_correct}/100 correctly classified")

    with (HERE / "instances.csv").open("w", newline="") as f:
        csv.writer(f).writerows(instances)
    print(f"wrote {len(instances)} instances to instances.csv")


if __name__ == "__main__":
    main()
