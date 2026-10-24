#!/bin/bash
P=~/Memloop; cd $P
export RUN_TAG=qwen3-14b-det
LOG=logs/supp_det.log
ARGS="--propagate --inject-num 5 --num-benign 20 --test-num 15 --eval-num 10"
run() { scripts/run_minja_recovery.sh $IDX 0 "$@" $ARGS; }
pair() {
  local IDX=$1; export WEBSHOP_URL=http://localhost:$2/
  note() { echo "[$(date +%T)] pair $IDX $*" >> $LOG; }
  note start
  WORK_SUFFIX=_r2 run counterfactual --cf-tag counterfactual_r2 & a=$!
  run recover --variant all & b=$!
  wait $a $b; note "cf_r2 + recovered_all done"
  run compare --base counterfactual_r2
  run baselines --baselines CK
  run forget --base counterfactual_r2 & a=$!
  ( sleep 60; until [ -f $P/results/minja_recovery/$RUN_TAG/$TAG/seed0/forget_target.json ]; do sleep 30; done
    S=$(python3 -c "import json;print(json.load(open('$P/results/minja_recovery/$RUN_TAG/$TAG/seed0/forget_target.json'))['session'])")
    WORK_SUFFIX=_fg run counterfactual --cf-tag cf_forget --forget-session $S ) & b=$!
  wait $a $b; note "forget + cf_forget done"
  run compare_forget --base counterfactual_r2
  run eval --eval-on CK recovered_all
  note DONE
}
victim() { python3 -c "import json;print(json.load(open('$HOME/Memloop/baselines/MINJA/rap/victim_target_pair/victim_target.json'))[$1]['victim'].replace(' ','_'))"; }
( IDX=8; TAG=$(victim 8); pair 8 3000 ) &
( IDX=5; TAG=$(victim 5); pair 5 3001 ) &
( IDX=2; TAG=$(victim 2); pair 2 3002 ) &
wait
echo "[$(date +%T)] SUPP_DET_DONE" >> $LOG
