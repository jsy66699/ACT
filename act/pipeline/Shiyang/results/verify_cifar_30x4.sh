#!/usr/bin/env bash
set -e
cd "D:\九州大学 研一 材料\研究课题 神经网络验证\ICSE 2027\Test_code\ACT"
source /d/ProgramFile/Anaconda/etc/profile.d/conda.sh
conda activate act-py312

# cifar100_2024, all 200 instances, 4 arms x 30 trials x 60s.
# Mirrors verify_safenlp_30x4.sh so the two benchmarks can be compared, but on
# a network where the safenlp conclusion may not carry:
#   - coverage does NOT saturate here (59.9% / 65.0%, with 1761 / 1315
#     never-activated neurons), so hpgd_cov can actually steer, unlike safenlp
#     where it fell back to Gaussian noise on one group entirely and ran on
#     only 19% of calls on the other.
#   - the pattern space is far larger, so "which direction to mutate" has more
#     room to matter than it did on a 30->128->2 MLP.
#
# Synthesis splits the 200 into resnet_medium (B=100) and resnet_large (B=100).
#
# --repeat 30 runs the trials inside one process: the VNNLIB load and synthesis
# cost 125s against 120s of fuzzing per trial, so per-trial re-invocation would
# spend 3.5 of 7.6 hours re-reading the same ONNX files. Each repeat still
# builds a fresh ACTFuzzer, verified by distinct hits and seeds_explored
# differing per repeat while coverage stays flat instead of accumulating.
#
# Caveat on resolution: a single trial hits only 1-2 distinct instances here
# (safenlp: 26-127), because few cifar100 instances are reachable at all --
# verify_large_trials_v2 found 6-7 over 60 trials on indices 100-199. The
# metric has a low ceiling; a safenlp-sized effect (4.6x) would still show, a
# small one may not.
OUT_REL=results/verify_cifar_30x4
OUT_ABS=act/pipeline/Shiyang/$OUT_REL
mkdir -p "$OUT_ABS"

run_arm () {
  local arm="$1"; shift
  echo "=== ${arm} x30 $(date) ==="
  python -m act.pipeline.Shiyang.pipeline.paper_cifar100_batch_ani \
    --category cifar100_2024 --max-instances 200 \
    --timeout 60 --repeat 30 --device cpu --no-save \
    "$@" --output "$OUT_REL/${arm}" \
    > "$OUT_ABS/${arm}.log" 2>&1
}

run_arm baseline  --hpgd-cov-weight 0
run_arm hpgdcov   --hpgd-cov-weight 0.5
run_arm statebase --hpgd-cov-weight 0   --admission-mode state
run_arm statehpgd --hpgd-weight 0.5     --admission-mode state

echo "ALL 120 CIFAR TRIALS DONE $(date)"
