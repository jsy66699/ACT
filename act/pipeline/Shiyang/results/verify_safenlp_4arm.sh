#!/usr/bin/env bash
set -e
cd "D:\九州大学 研一 材料\研究课题 神经网络验证\ICSE 2027\Test_code\ACT"
source /d/ProgramFile/Anaconda/etc/profile.d/conda.sh
conda activate act-py312

# safenlp_2024, all 1080 instances, 4 arms x 10 trials x 60s, on 6696a18.
# Everything is re-run on one code version: 6696a18 changed state admission
# (fingerprint path + per-instance state space), so no earlier state-mode
# result is comparable, and running some arms on older code would hand them a
# different iteration budget for the same wall clock.
#
#   arm         strategies                      admission   isolates
#   baseline    pgd 50 / bnd 20 / rnd 30        coverage    reference
#   hpgdcov     + hpgd_cov 33.3%, pgd 50        coverage    coverage-guided mutation
#   statebase   pgd 50 / bnd 20 / rnd 30        state       state admission ALONE
#   statehpgd   + hpgd 33.3%, pgd 50            state       both together
#
# pgd is pinned at 50% in all four (build_mutation_weights displaces from
# boundary/random, never from pgd), so CE-seeking budget is matched and the
# only moved variable per comparison is the one named above.
#
# Read as: baseline->statebase is state admission's own contribution,
# statebase->statehpgd is what the hpgd strategy adds on top of it, and
# baseline->hpgdcov is the coverage-guided equivalent for contrast.
#
# Metric is distinct_instances_hit, NOT violations: on this benchmark one
# instance yields thousands of near-duplicate counterexamples, so raw CE count
# mostly measures how long the corpus snowballed on a few instances.
OUT_REL=results/verify_safenlp_4arm
OUT_ABS=act/pipeline/Shiyang/$OUT_REL
mkdir -p "$OUT_ABS"

run_arm () {   # $1 = arm name, rest = extra flags
  local arm="$1"; shift
  for i in $(seq 1 10); do
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

echo "ALL 40 TRIALS DONE $(date)"
