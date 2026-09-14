# ERAN Sigmoid / Tanh MNIST MLPs

Six fully-connected MNIST networks with smooth activations — the standard
Sigmoid/Tanh benchmark set used by DeepPoly, PRIMA, VeriNet, α,β-CROWN and
GenBaB.

| model | activation | architecture | epsilon | instances |
|---|---|---|---|---|
| `ffnnSIGMOID__Point_6x100.onnx` | Sigmoid | 6 × 100 | 0.015 | 99 |
| `ffnnSIGMOID__Point_6x200.onnx` | Sigmoid | 6 × 200 | 0.012 | 99 |
| `ffnnSIGMOID__Point_9x100.onnx` | Sigmoid | 9 × 100 | 0.015 | 99 |
| `ffnnTANH__Point_6x100.onnx` | Tanh | 6 × 100 | 0.006 | 97 |
| `ffnnTANH__Point_6x200.onnx` | Tanh | 6 × 200 | 0.002 | 98 |
| `ffnnTANH__Point_9x100.onnx` | Tanh | 9 × 100 | 0.006 | 98 |

590 instances total, timeout 300 s.

## Provenance

Models: <https://huggingface.co/datasets/zhouxingshi/GenBaB> (`eran/*`), which
repackages the ETH SRI ERAN pretrained nets. Epsilons are the ones in the
GenBaB `config.yaml` files, i.e. the settings behind the numbers in
*Neural Network Verification with Branch-and-Bound for General Nonlinearities*
(Shi et al., TACAS 2025, <https://arxiv.org/abs/2405.21063>).

The upstream release ships no VNNLIB specs (α,β-CROWN generates them from the
dataset config), so `generate_specs.py` writes them here: L-inf balls of radius
epsilon around the first 100 MNIST test images (`data/csv/mnist_first_100_samples.csv`),
clipped to [0, 1], skipping images the model misclassifies. Specs are VNNLIB 2.0
(the only form ACT reads); use `tools/convert_vnnlib_2_to_1.py` for 1.0-only
verifiers such as α,β-CROWN.

## Two ONNX variants

* `onnx/` — as downloaded. MNIST normalization is baked into the graph as a
  leading `Sub(0.1307) → Div(0.3081)`, so inputs are raw [0, 1] pixels.
  **ACT's `torch2act` cannot build these**: the scalar-vs-784 broadcast raises
  `NotImplementedError: Var-var 'sub' size mismatch (784 vs 1)`.
* `onnx_folded/` — same networks with that normalization folded into the first
  Gemm (`W/s`, `b − W·m/s`), produced by `fold_normalization.py`. Inputs are
  still raw [0, 1] pixels, so the specs are shared; outputs match the originals
  to ≤ 7.4e-05 over the 100 test images (float32 round-off). All six build into
  ACT nets with `SIGMOID` / `TANH` layers.

`instances.csv` points at `onnx_folded/` so the category loads in ACT out of the
box; `instances_original.csv` points at `onnx/` for verifiers that handle the
in-graph normalization (α,β-CROWN among them).

## Regenerating

```bash
python data/vnnlib/eran_sigmoid_tanh_mlp/generate_specs.py      # vnnlib/ + instances.csv
python data/vnnlib/eran_sigmoid_tanh_mlp/fold_normalization.py  # onnx_folded/ + instances_folded.csv
# then: mv instances.csv instances_original.csv; mv instances_folded.csv instances.csv
```
