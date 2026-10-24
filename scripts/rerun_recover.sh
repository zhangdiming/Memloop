#!/bin/bash
P=~/Memloop; cd $P
LOG=logs/pilot_recovery.log
ARGS="--propagate --inject-num 5 --num-benign 20 --test-num 15 --eval-num 10"
one() {
  export WEBSHOP_URL=http://localhost:$2/
  R=results/minja_recovery/qwen3-14b/$(python3 -c "import json;print(json.load(open('baselines/MINJA/rap/victim_target_pair/victim_target.json'))[$1]['victim'].replace(' ','_'))")/seed0
  mkdir -p $R/stale_recover && mv $R/recover*.log $R/stale_recover/ 2>/dev/null
  scripts/run_minja_recovery.sh $1 0 recover $ARGS && echo "[$(date +%T)] pair $1 recover (fixed) done" >> $LOG
  scripts/run_minja_recovery.sh $1 0 eval $ARGS --eval-on recovered && echo "[$(date +%T)] pair $1 eval recovered done" >> $LOG
}
one 8 3000 & one 5 3001 & one 2 3002 &
wait
echo "[$(date +%T)] RERUN_RECOVER_DONE" >> $LOG
