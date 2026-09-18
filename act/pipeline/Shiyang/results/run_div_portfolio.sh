#!/usr/bin/env bash
# CE-diversity dump runs for the three PORTFOLIO arms: baseline / statebase /
# statehpgd -- the arms that vary only admission and the mutation portfolio,
# with every energy switch (ce_energy, ce-parent-replacement) left at its
# published default. Rebuilt 2026-09-17 from the 2026-09-07 scratchpad version.
#
#   bash act/pipeline/Shiyang/results/run_div_portfolio.sh            # cifar
#   BENCH=cora bash .../run_div_portfolio.sh
#   BENCH=all  bash .../run_div_portfolio.sh
#
# Writes results/div_<bench>/<arm>/<group>/ce_sample.npz, then analyses them
# with ce_diversity.py --arms portfolio, which prints per arm:
#
#   D  instances broken                  (inter-instance diversity)
#   S  mean pairwise cosine distance     (intra-instance diversity)
#
# WHAT THIS SCRIPT'S D IS NOT. Each arm runs ONCE here, and a single run
# cannot rank arms: on safenlp the campaign has statebase 36.60 > statehpgd
# 32.20 (n=5) while one run gave 30 vs 37; on cifar the campaign is 6.77 vs
# 6.37 (n=30) and one run gave 5 vs 9. The citable distinct numbers come from
# the repeat-30 campaign (verify_cifar_30x4*.sh and its siblings) and are
# already baked into ce_diversity.py's CAMPAIGN_D, which is what the figure's
# D axis reads. Use this script for S; use the campaign for distinct.
#
# --no-save IS LOAD-BEARING. Without it a 10-second exploratory run wrote
# 7,097 ce_*.pt files totalling 2.6 GB. The existing div_* directories contain
# zero .pt files, which is the evidence it was set.
set -uo pipefail

ACT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ACT_ROOT"

BENCH="${BENCH:-cifar}"
TIMEOUT="${TIMEOUT:-60}"
DEVICE="${DEVICE:-cpu}"
CLUSTERS="${CLUSTERS:-250}"
FORCE="${FORCE:-0}"
ANALYSE="${ANALYSE:-1}"

# bench -> "output_dir category max_instances [extra driver flags]".
# Mirrors ce_diversity.py's BENCH table; the two eran entries pin a single
# model group by index because the category holds six nets and a "first N
# specs" prefix cannot isolate the tanh ones.
bench_cfg () {
  case "$1" in
    cora)      echo "div_cora cora_2024 180" ;;
    mnist)     echo "div_mnist mnist_fc_v2 90" ;;
    cifar)     echo "div_cifar cifar100_2024 200" ;;
    safenlp)   echo "div_safenlp safenlp_2024 200" ;;
    eransig)   echo "div_eran_sig6x100 eran_sigmoid_tanh_mlp 99" ;;
    erantanh)  echo "div_eran_tanh6x100 eran_sigmoid_tanh_mlp 394" ;;
    *)         echo "" ;;
  esac
}

ALL_BENCHES="cora mnist cifar safenlp eransig erantanh"
if [ "$BENCH" = "all" ]; then
  TODO="$ALL_BENCHES"
else
  TODO="$BENCH"
fi

for b in $TODO; do
  if [ -z "$(bench_cfg "$b")" ]; then
    echo "unknown BENCH=$b; expected one of: $ALL_BENCHES all"; exit 2
  fi
done

run_arm () {
  local bench="$1" dir="$2" category="$3" max_inst="$4" arm="$5"; shift 5
  local out_rel="results/${dir}/${arm}"
  local out_abs="$ACT_ROOT/act/pipeline/Shiyang/${out_rel}"

  if [ -d "$out_abs" ] && [ -n "$(ls -A "$out_abs" 2>/dev/null)" ]; then
    if [ "$FORCE" != "1" ]; then
      echo "=== ${bench}/${arm} SKIPPED: $out_rel already has results (FORCE=1 to overwrite) ==="
      return 0
    fi
    echo "=== ${bench}/${arm} FORCE: removing $out_rel ==="
    rm -rf "$out_abs"
  fi
  mkdir -p "$out_abs"

  echo "=== ${bench}/${arm} START $(date '+%H:%M:%S') ==="
  python -u -m act.pipeline.Shiyang.pipeline.paper_cifar100_batch_ani \
    --category "$category" --max-instances "$max_inst" \
    --timeout "$TIMEOUT" --repeat 1 --device "$DEVICE" --no-save \
    --dump-founder-clusters "$CLUSTERS" \
    "$@" --output "$out_rel" \
    > "$out_abs/../${arm}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "!!! ${bench}/${arm} FAILED rc=$rc -- tail of log:"
    tail -20 "$out_abs/../${arm}.log"
  else
    echo "=== ${bench}/${arm} DONE  $(date '+%H:%M:%S') ==="
  fi
  return $rc
}

failed=""
for b in $TODO; do
  read -r dir category max_inst <<< "$(bench_cfg "$b")"
  echo
  echo "########## $b  ($category, $max_inst specs, ${TIMEOUT}s/group, $DEVICE) ##########"

  # The three arms differ from each other in exactly one thing at a time:
  # baseline -> statebase adds state admission; statebase -> statehpgd adds
  # HPGD to the portfolio. HPGD requires state admission (driver guard,
  # paper_cifar100_batch_ani.py:313), so statehpgd carries both flags and its
  # control is statebase, not baseline.
  run_arm "$b" "$dir" "$category" "$max_inst" baseline  || failed="$failed $b/baseline"
  run_arm "$b" "$dir" "$category" "$max_inst" statebase --admission-mode state \
    || failed="$failed $b/statebase"
  run_arm "$b" "$dir" "$category" "$max_inst" statehpgd --admission-mode state --hpgd-weight 0.5 \
    || failed="$failed $b/statehpgd"
done

echo
if [ -n "$failed" ]; then
  echo "=== RUNS FAILED:$failed"
  echo "=== skipping analysis"
  exit 1
fi
echo "=== ALL DUMP RUNS DONE $(date) ==="

if [ "$ANALYSE" != "1" ]; then
  echo "ANALYSE=0, stopping before ce_diversity."
  exit 0
fi

echo
echo "########## ce_diversity --arms portfolio ##########"
if [ "$BENCH" = "all" ]; then
  python -u -m act.pipeline.Shiyang.pipeline.ce_diversity --arms portfolio all
else
  python -u -m act.pipeline.Shiyang.pipeline.ce_diversity --arms portfolio "$BENCH"
fi
echo
echo "Figures: act/pipeline/Shiyang/results/fig_ce_diversity_3arm_portfolio_*.png"
