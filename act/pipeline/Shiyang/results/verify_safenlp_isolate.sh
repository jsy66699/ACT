#!/usr/bin/env bash
set -e
cd "D:\九州大学 研一 材料\研究课题 神经网络验证\ICSE 2027\Test_code\ACT"
source /d/ProgramFile/Anaconda/etc/profile.d/conda.sh
conda activate act-py312

# Isolates what the 4.4x distinct-instance gain in verify_safenlp_10x3's
# statehpgd arm actually came from. That arm moved TWO variables at once
# against baseline: it added the hpgd strategy AND switched admission_mode
# from coverage to state.
#
#   statebase  = baseline weights (pgd 50 / boundary 20 / random 30)
#                + admission_mode state          <- state admission ALONE
#   statehpgd  = + hpgd 33.3%, pgd still 50%
#                + admission_mode state          <- both, as before
#
# statehpgd is re-run rather than reused because commit 8bad6b3 sped up state
# admission by ~32% (exact-match set instead of a BK-tree walk). Only the state
# path changed, so verify_safenlp_10x3's baseline and hpgdcov arms are still
# valid and are NOT re-run; but comparing a new-code statebase against an
# old-code statehpgd would hand statebase a third more iterations for free.
#
# Reading:
#   statebase ~= statehpgd  -> the gain is state admission; hpgd adds little
#   statebase ~= baseline   -> the gain needs the hpgd strategy (or the pairing)
OUT_REL=results/verify_safenlp_isolate
OUT_ABS=act/pipeline/Shiyang/$OUT_REL
mkdir -p "$OUT_ABS"

for i in $(seq 1 10); do
  echo "=== statebase_s${i} $(date) ==="
  python -m act.pipeline.Shiyang.pipeline.paper_cifar100_batch_ani \
    --category safenlp_2024 --max-instances 1080 \
    --hpgd-cov-weight 0 --admission-mode state --timeout 60 --device cpu \
    --no-save --output "$OUT_REL/statebase_s${i}" \
    > "$OUT_ABS/statebase_s${i}.log" 2>&1
done

for i in $(seq 1 10); do
  echo "=== statehpgd_s${i} $(date) ==="
  python -m act.pipeline.Shiyang.pipeline.paper_cifar100_batch_ani \
    --category safenlp_2024 --max-instances 1080 \
    --hpgd-weight 0.5 --admission-mode state --timeout 60 --device cpu \
    --no-save --output "$OUT_REL/statehpgd_s${i}" \
    > "$OUT_ABS/statehpgd_s${i}.log" 2>&1
done

echo "ALL 20 ISOLATION TRIALS DONE $(date)"
