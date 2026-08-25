#!/usr/bin/env bash
set -e
cd "D:\九州大学 研一 材料\研究课题 神经网络验证\ICSE 2027\Test_code\ACT"
source /d/ProgramFile/Anaconda/etc/profile.d/conda.sh
conda activate act-py312

# safenlp_2024, all 1080 instances, 3 strategies x 10 trials x 60s.
# Synthesis splits these into 2 groups (medical B=275 / ruarobot B=805), so
# each trial is 2 x 60s + ~16s load = ~136s; 30 trials = ~68 min.
#
# Caveat, measured (not assumed) -- and it differs BY GROUP, so read the
# per-group "HPGD-Cov calls:" line each trial now prints rather than treating
# the arm as uniform:
#   ruarobot: GlobalCov hits 100%, 0 never-activated neurons -> HPGDCoverageMutation
#     finds nothing to chase and takes its "fully covered -> Gaussian noise"
#     fallback for the entire run. This group's hpgdcov arm is really
#     "boundary+random partly replaced by random noise", NOT a test of
#     coverage-targeted mutation.
#   medical: GlobalCov plateaus at 99.22% with 2 never-activated neurons, so
#     hpgd_cov steers on every call. This group's comparison IS meaningful.
# baseline vs statev2 is unaffected in both, since admission_mode does not
# touch coverage.
#
# Weights use the displacing scheme (build_mutation_weights): pgd stays at
# 50% in all three arms, so CE-seeking budget is matched.
OUT_REL=results/verify_safenlp_10x3
OUT_ABS=act/pipeline/Shiyang/$OUT_REL
mkdir -p "$OUT_ABS"

for i in $(seq 1 10); do
  echo "=== baseline_s${i} $(date) ==="
  python -m act.pipeline.Shiyang.pipeline.paper_cifar100_batch_ani \
    --category safenlp_2024 --max-instances 1080 \
    --hpgd-cov-weight 0 --timeout 60 --device cpu \
    --no-save --output "$OUT_REL/baseline_s${i}" \
    > "$OUT_ABS/baseline_s${i}.log" 2>&1
done

for i in $(seq 1 10); do
  echo "=== hpgdcov_s${i} $(date) ==="
  python -m act.pipeline.Shiyang.pipeline.paper_cifar100_batch_ani \
    --category safenlp_2024 --max-instances 1080 \
    --hpgd-cov-weight 0.5 --timeout 60 --device cpu \
    --no-save --output "$OUT_REL/hpgdcov_s${i}" \
    > "$OUT_ABS/hpgdcov_s${i}.log" 2>&1
done

# hpgd (NOT hpgd_cov) here, deliberately. HPGDMutation flips ReLU signs inside
# PatternStateManager's unstable subspace, which is the same subspace state
# admission scores novelty over -- strategy and admission agree on the signal.
# Pairing hpgd_cov with state instead (what verify_large_trials_v2's state_v2
# arm did) crosses two different neuron spaces: the strategy hunts uncovered
# neurons in CoverageTracker's output space while admission judges ReLU-sign
# novelty, so lighting up a new neuron does not help the sample survive.
# Both guided strategies take the identical 33.3% displaced share, so the cov
# and state arms differ only in which one runs.
for i in $(seq 1 10); do
  echo "=== statehpgd_s${i} $(date) ==="
  python -m act.pipeline.Shiyang.pipeline.paper_cifar100_batch_ani \
    --category safenlp_2024 --max-instances 1080 \
    --hpgd-weight 0.5 --admission-mode state --timeout 60 --device cpu \
    --no-save --output "$OUT_REL/statehpgd_s${i}" \
    > "$OUT_ABS/statehpgd_s${i}.log" 2>&1
done

echo "ALL 30 SAFENLP TRIALS DONE $(date)"
