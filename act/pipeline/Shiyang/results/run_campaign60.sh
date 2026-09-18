#!/usr/bin/env bash
# Full campaign: every benchmark, three portfolio arms, 60 s per group,
# REPEAT trials, with the CE-diversity dumps switched on -- so one pass yields
# BOTH numbers:
#
#   distinct  mean +- sd over REPEAT trials, plus the union, from
#             <arm>/rep*/<group>/group_summary.json   (summarize_state_ab.py)
#   S         mean pairwise cosine distance between counterexample constraint
#             directions, from <arm>/<group>/ce_sample.npz   (ce_diversity.py)
#
#   bash act/pipeline/Shiyang/results/run_campaign60.sh                  # all six
#   BENCHES="cifar mnist" bash .../run_campaign60.sh                     # a slice
#   REPEAT=5 bash .../run_campaign60.sh                                  # rehearsal
#
# COST. 21 model groups per trial across the six benchmarks (cora 9, erantanh
# 4, mnist 3, cifar 2, safenlp 2, eransig 1), so one arm-trial is 21 minutes of
# fuzzing and REPEAT=60 x 3 arms is about 63 hours SEQUENTIAL. --repeat runs
# the trials inside one process, so the ONNX load and model build are paid once
# per arm, not once per trial. Split with BENCHES across several tmux windows
# to parallelise; the six are independent and write to disjoint directories.
# Wall-clock floor is then cora's 27 h.
#
# RESUME. An arm whose rep<REPEAT> directory already exists is skipped, so a
# killed run is restarted by re-invoking the same command. A PARTIAL arm (some
# reps present, not all) is NOT resumed -- the driver always starts at rep1 --
# so it is reported and left alone; delete that arm directory to redo it, or
# raise REPEAT and treat the reps you have as a smaller n.
#
# WHAT THE DUMPS DO AND DO NOT GIVE YOU. With --repeat > 1 the driver writes
# each trial under rep<k>/, so the dumps land at <arm>/rep<k>/<group>/
# ce_sample.npz while ce_diversity.py reads <arm>/<group>/ce_sample.npz. This
# script symlinks rep1's group directories up to the arm root to satisfy it.
# S is therefore still measured on ONE trial, exactly as the published n=1
# diversity runs were -- REPEAT buys n for distinct, not for S. Aggregating S
# over trials would need a change to ce_diversity.py, which this script does
# not make.
#
# --no-save IS LOAD-BEARING: without it a 10-second run wrote 7,097 ce_*.pt
# totalling 2.6 GB. With it, expect ~4 GB of ce_sample.npz for a full pass.
set -uo pipefail

ACT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ACT_ROOT"
RES="$ACT_ROOT/act/pipeline/Shiyang/results"

REPEAT="${REPEAT:-60}"
TIMEOUT="${TIMEOUT:-60}"
DEVICE="${DEVICE:-cpu}"
CLUSTERS="${CLUSTERS:-250}"
BENCHES="${BENCHES:-cora mnist cifar safenlp eransig erantanh}"
ANALYSE="${ANALYSE:-1}"
# Flags appended to EVERY arm, so whatever they switch on is held CONSTANT
# across the three and the arms still isolate admission and guidance. Putting
# a flag on only some arms is what invalidated the original four-arm set --
# its state and hpgd arms each also carried ce_energy 1 + cerepl, so neither
# could be read as "what state admission is worth" (ce_diversity.py ARM_SETS).
# The 2026-09-18 pass runs EXTRA="--ce-parent-replacement".
EXTRA="${EXTRA:-}"

# bench -> "output_dir category max_instances". Mirrors ce_diversity.py's BENCH
# table, so both analyses read the directories this writes.
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

for b in $BENCHES; do
  [ -n "$(bench_cfg "$b")" ] || { echo "unknown benchmark: $b"; exit 2; }
done

echo "campaign: benches=[$BENCHES] arms=[baseline statebase statehpgd]"
echo "          repeat=$REPEAT timeout=${TIMEOUT}s device=$DEVICE clusters=$CLUSTERS"
echo "          every arm also gets: ${EXTRA:-(nothing)}"
echo "          started $(date)"

skipped=""
failed=""
partial=""

run_arm () {
  local bench="$1" dir="$2" category="$3" max_inst="$4" arm="$5"; shift 5
  local out_rel="results/${dir}/${arm}"
  local out_abs="${RES}/${dir}/${arm}"

  if [ -d "$out_abs/rep${REPEAT}" ]; then
    echo "=== ${bench}/${arm} already complete (rep${REPEAT} present), skipping ==="
    skipped="$skipped ${bench}/${arm}"
    return 0
  fi
  if [ -d "$out_abs" ] && [ -n "$(ls -A "$out_abs" 2>/dev/null)" ]; then
    local have
    have=$(find "$out_abs" -maxdepth 1 -name 'rep*' -type d 2>/dev/null | wc -l)
    echo "!!! ${bench}/${arm} is PARTIAL (${have}/${REPEAT} reps) -- not resumable, leaving it alone."
    echo "    rm -rf $out_rel   to redo it from scratch."
    partial="$partial ${bench}/${arm}(${have})"
    return 0
  fi
  mkdir -p "$out_abs"

  echo "=== ${bench}/${arm} START $(date '+%F %H:%M:%S') ==="
  python -u -m act.pipeline.Shiyang.pipeline.paper_cifar100_batch_ani \
    --category "$category" --max-instances "$max_inst" \
    --timeout "$TIMEOUT" --repeat "$REPEAT" --device "$DEVICE" --no-save \
    --dump-founder-clusters "$CLUSTERS" \
    $EXTRA "$@" --output "$out_rel" \
    > "${RES}/${dir}/${arm}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "!!! ${bench}/${arm} FAILED rc=$rc $(date '+%F %H:%M:%S') -- tail:"
    tail -20 "${RES}/${dir}/${arm}.log"
    failed="$failed ${bench}/${arm}"
    return $rc
  fi
  echo "=== ${bench}/${arm} DONE  $(date '+%F %H:%M:%S') ==="

  # ce_diversity.py reads <arm>/<group>/ce_sample.npz; --repeat put it at
  # <arm>/rep1/<group>/. Link, do not copy: the npz files reach ~1.7 MB each.
  local n=0
  for g in "$out_abs"/rep1/*/; do
    [ -f "${g}ce_sample.npz" ] || continue
    ln -sfn "$g" "$out_abs/$(basename "$g")" && n=$((n + 1))
  done
  echo "    linked $n group dump(s) from rep1 to the arm root for ce_diversity"
  return 0
}

for b in $BENCHES; do
  read -r dir category max_inst <<< "$(bench_cfg "$b")"
  echo
  echo "##################################################################"
  echo "# $b  ($category, $max_inst specs, ${TIMEOUT}s/group x $REPEAT trials)"
  echo "##################################################################"

  # One variable moves at a time: baseline -> statebase adds state admission,
  # statebase -> statehpgd adds HPGD. HPGD requires state admission (driver
  # guard, paper_cifar100_batch_ani.py:313), so statehpgd's control is
  # statebase, never baseline.
  run_arm "$b" "$dir" "$category" "$max_inst" baseline
  run_arm "$b" "$dir" "$category" "$max_inst" statebase --admission-mode state
  run_arm "$b" "$dir" "$category" "$max_inst" statehpgd --admission-mode state --hpgd-weight 0.5
done

echo
echo "=== CAMPAIGN RUNS FINISHED $(date) ==="
[ -n "$skipped" ] && echo "    skipped (already complete):$skipped"
[ -n "$partial" ] && echo "    PARTIAL, not rerun:$partial"
[ -n "$failed" ]  && echo "    FAILED:$failed"

if [ "$ANALYSE" != "1" ]; then
  echo "ANALYSE=0, stopping before the summaries."
  exit 0
fi
if [ -n "$failed" ]; then
  echo "skipping analysis because some arms failed"
  exit 1
fi

echo
echo "##################################################################"
echo "# distinct  (mean +- sd over trials, and the union)"
echo "##################################################################"
for b in $BENCHES; do
  read -r dir _ _ <<< "$(bench_cfg "$b")"
  echo
  echo "--- $b ($dir) ---"
  python -u -m act.pipeline.Shiyang.pipeline.summarize_state_ab "$dir"
done

echo
echo "##################################################################"
echo "# diversity  (D from one trial -- do not rank arms on it -- and S)"
echo "##################################################################"
python -u -m act.pipeline.Shiyang.pipeline.ce_diversity --arms portfolio all

echo
echo "Figures:  $RES/fig_ce_diversity_3arm_portfolio_*.png"
echo "NOTE: ce_diversity's D axis reads the hard-coded CAMPAIGN_D table in"
echo "      ce_diversity.py, NOT the run you just did. Update CAMPAIGN_D with"
echo "      the distinct means printed above before citing the figure."
echo "DONE $(date)"
