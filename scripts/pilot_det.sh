#!/bin/bash
P=~/Memloop; cd $P
export RUN_TAG=qwen3-14b-det
LOG=logs/pilot_det.log
ARGS="--propagate --inject-num 5 --num-benign 20 --test-num 15 --eval-num 10"
pair() {
  local IDX=$1; export WEBSHOP_URL=http://localhost:$2/
  scripts/run_minja_recovery.sh $IDX 0 attack $ARGS || { echo "[$(date +%T)] pair $IDX attack FAILED" >> $LOG; return; }
  echo "[$(date +%T)] pair $IDX attack done" >> $LOG
  scripts/run_minja_recovery.sh $IDX 0 recover $ARGS & a=$!
  scripts/run_minja_recovery.sh $IDX 0 counterfactual $ARGS & b=$!
  scripts/run_minja_recovery.sh $IDX 0 baselines $ARGS & c=$!
  wait $a $b $c
  echo "[$(date +%T)] pair $IDX recover/counterfactual/baselines done" >> $LOG
  scripts/run_minja_recovery.sh $IDX 0 eval $ARGS
  scripts/run_minja_recovery.sh $IDX 0 compare $ARGS
  echo "[$(date +%T)] pair $IDX eval+compare done" >> $LOG
}
pair 8 3000 & pair 5 3001 & pair 2 3002 &
wait
echo "[$(date +%T)] PILOT_DET_DONE" >> $LOG
