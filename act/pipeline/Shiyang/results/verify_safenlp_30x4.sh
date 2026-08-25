#!/usr/bin/env bash
set -e
cd "D:\九州大学 研一 材料\研究课题 神经网络验证\ICSE 2027\Test_code\ACT"
source /d/ProgramFile/Anaconda/etc/profile.d/conda.sh
conda activate act-py312

# safenlp_2024, all 1080 instances, 4 arms x 30 trials x 60s, on 66280e9.
# ~120 trials x ~136s = ~4.5h.
#
#   arm         strategies                   admission   isolates
#   baseline    pgd 50 / bnd 20 / rnd 30     coverage    reference
#   hpgdcov     + hpgd_cov 33.3%, pgd 50     coverage    coverage-guided mutation
#   statebase   pgd 50 / bnd 20 / rnd 30     state       state admission ALONE
#   statehpgd   + hpgd 33.3%, pgd 50         state       + pattern-guided mutation
#
# n=30 rather than the previous 10 because the quantity being compared is
# noisy: statehpgd measured 120.50 +- 37.86 distinct instances at n=10, a 31%
# coefficient of variation, which left the statebase -> statehpgd increment
# (+14.40, t=+0.92) indistinguishable from zero. n=30 shrinks the standard
# error by ~1.7x, enough to resolve an effect of that size if it is real.
#
# statehpgd is not the same arm it was in verify_safenlp_4arm: 66280e9
# connected HPGD's local_bias feedback, restricted its hinge loss to the
# neurons it asks to flip, and steers its target choice by per-neuron marginal
# occupancy. All four arms run on that one commit; earlier numbers are not
# comparable and are not reused.
#
# Metric is distinct_instances_hit, NOT violations: one instance yields
# thousands of near-duplicate counterexamples here, so raw CE count mostly
# measures how long the corpus snowballed on a few instances.
OUT_REL=results/verify_safenlp_30x4
OUT_ABS=act/pipeline/Shiyang/$OUT_REL
mkdir -p "$OUT_ABS"

run_arm () {
  local arm="$1"; shift
  for i in $(seq 1 30); do
    echo "=== ${arm}_s${i} $(date) ==="
    python -m act.pipeline.Shiyang.pipeline.paper_cifar100_batch_ani \
      --category safenlp_2024 --max-instances 1080 \
      --timeout 60 --device cpu --no-save \
      "$@" --output "$OUT_REL/${arm}_s${i}" \
      > "$OUT_ABS/${arm}_s${i}.log" 2>&1
  done
}

run_arm baseline  --hpgd-cov-weight 0
run_arm hpgdcov   --hpgd-cov-weight 0.5
run_arm statebase --hpgd-cov-weight 0   --admission-mode state
run_arm statehpgd --hpgd-weight 0.5     --admission-mode state

echo "ALL 120 TRIALS DONE $(date)"
