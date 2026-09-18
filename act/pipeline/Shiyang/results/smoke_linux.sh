#!/usr/bin/env bash
# Linux smoke test: verify the environment, the data, and one short fuzzing run.
#
# Unlike the other scripts in this directory, this one hardcodes no paths --
# it locates the repo root from its own location, so it works wherever the
# checkout lives. Run it from anywhere:
#
#   conda activate act-py312
#   bash act/pipeline/Shiyang/results/smoke_linux.sh
#
# Override the defaults with env vars, e.g.
#   CATEGORY=cora_2024 MAX_INSTANCES=20 TIMEOUT=60 bash .../smoke_linux.sh
set -uo pipefail

ACT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ACT_ROOT"

CATEGORY="${CATEGORY:-mnist_fc_v2}"
MAX_INSTANCES="${MAX_INSTANCES:-10}"
TIMEOUT="${TIMEOUT:-20}"
DEVICE="${DEVICE:-cpu}"
OUT_REL="${OUT_REL:-results/_linux_smoke/first_try}"
OUT_ABS="$ACT_ROOT/act/pipeline/Shiyang/$OUT_REL"

echo "=== repo root: $ACT_ROOT"
echo "=== python:    $(command -v python)"

fail=0

echo
echo "--- 1. environment ---"
python - <<'PY' || fail=1
import sys
print("python  ", sys.version.split()[0])
try:
    import torch
    print("torch   ", torch.__version__, "cuda:", torch.cuda.is_available())
except Exception as e:
    print("torch    MISSING:", e); raise SystemExit(1)
for mod in ("onnx", "onnx2torch", "onnxruntime", "numpy", "pandas", "networkx", "yaml", "psutil"):
    try:
        __import__(mod)
        print(f"{mod:<9} ok")
    except Exception as e:
        print(f"{mod:<9} MISSING: {e}"); raise SystemExit(1)
PY

echo
echo "--- 2. act package imports ---"
python -c "import act; from act.pipeline.fuzzing import pattern_search_pgd; print('act ok')" || fail=1

echo
echo "--- 3. benchmark data ---"
for bench in cora_2024 mnist_fc_v2 cifar100_2024 safenlp_2024 eran_sigmoid_tanh_mlp; do
  d="data/vnnlib/$bench"
  if [ ! -d "$d" ]; then
    echo "$bench  MISSING directory"; fail=1; continue
  fi
  if [ ! -f "$d/instances.csv" ]; then
    echo "$bench  MISSING instances.csv"; fail=1; continue
  fi
  n=$(wc -l < "$d/instances.csv")
  echo "$(printf '%-24s' "$bench") $(du -sh "$d" | cut -f1)  instances.csv: ${n} lines"
done
if [ -d data/vnnlib/eran_sigmoid_tanh_mlp ] && [ ! -d data/vnnlib/eran_sigmoid_tanh_mlp/onnx_folded ]; then
  echo "WARNING: eran_sigmoid_tanh_mlp/onnx_folded missing -- the smooth-activation"
  echo "         runs need the folded ONNX (torch2act cannot build in-graph Sub/Div)."
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  echo
  echo "=== PRE-FLIGHT FAILED -- fix the above before running the fuzzer ==="
  exit 1
fi

echo
echo "--- 4. short fuzzing run: $CATEGORY, $MAX_INSTANCES instances, ${TIMEOUT}s, device=$DEVICE ---"
mkdir -p "$OUT_ABS"
LOG="$OUT_ABS/smoke.log"
set -x
python -u -m act.pipeline.Shiyang.pipeline.paper_cifar100_batch_ani \
  --category "$CATEGORY" --max-instances "$MAX_INSTANCES" \
  --timeout "$TIMEOUT" --repeat 1 --device "$DEVICE" --no-save \
  --output "$OUT_REL" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set +x

echo
if [ "$rc" -eq 0 ]; then
  echo "=== SMOKE OK (exit 0). Full log: $LOG"
  ls -la "$OUT_ABS"
else
  echo "=== SMOKE FAILED (exit $rc). Last 40 lines:"
  tail -40 "$LOG"
fi
exit "$rc"
