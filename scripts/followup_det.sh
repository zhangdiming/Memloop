#!/bin/bash
P=~/Memloop; cd $P
export RUN_TAG=qwen3-14b-det
LOG=logs/pilot_det.log
until grep -q PILOT_DET_DONE $LOG 2>/dev/null; do sleep 120; done
ARGS="--propagate --inject-num 5 --num-benign 20 --test-num 15 --eval-num 10"
pair() {
  local IDX=$1; export WEBSHOP_URL=http://localhost:$2/
  scripts/run_minja_recovery.sh $IDX 0 recover $ARGS --variant b3 & a=$!
  scripts/run_minja_recovery.sh $IDX 0 recover $ARGS --variant contain & b=$!
  wait $a $b
  echo "[$(date +%T)] pair $IDX b3+contain done" >> $LOG
  scripts/run_minja_recovery.sh $IDX 0 eval $ARGS --eval-on recovered_b3 recovered_contain
  scripts/run_minja_recovery.sh $IDX 0 compare $ARGS
  echo "[$(date +%T)] pair $IDX followup eval+compare done" >> $LOG
}
pair 8 3000 & pair 5 3001 & pair 2 3002 &
wait
echo "[$(date +%T)] FOLLOWUP_DET_DONE" >> $LOG
