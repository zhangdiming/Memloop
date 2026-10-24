#!/bin/bash
IDX=$1; SEED=$2
P=~/Memloop; cd $P
LOG=logs/alfworld_sweep.log
R() { scripts/alfworld_run.sh $IDX $SEED "$@"; }
V=$(python3 -c "import json;print(json.load(open('memloop/experiments/alfworld_pairs.json'))[$IDX]['victim'])")
D=results/alfworld/${RUN_TAG:-qwen3-14b-main}/$V/seed$SEED
note() { echo "[$(date +%T)] $V/$SEED $*" >> $LOG; }
note start
[ -f $D/attack.pkl ] || R attack || { note "attack failed"; exit 1; }
pids=()
[ -f $D/counterfactual.pkl ] || { R counterfactual & pids+=($!); }
[ -f $D/recovered_contain.pkl ] || { WORK_SUFFIX=_contain R recover --variant contain & pids+=($!); }
[ -f $D/recovered_b3.pkl ] || { WORK_SUFFIX=_b3 R recover --variant b3 & pids+=($!); }
for p in "${pids[@]}"; do wait $p || note "a phase failed (pid $p)"; done
R baselines --baselines B1 B2 DP
E=$D/eval_attack_B1_B2_DP_recovered_contain_recovered_b3_counterfactual.json
[ -f $E ] || R eval --eval-on attack B1 B2 DP recovered_contain recovered_b3 counterfactual
R compare
[ -f $E ] && note "done" || note "eval missing"
