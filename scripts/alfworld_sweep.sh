#!/bin/bash
P=~/Memloop; cd $P
MAXJ=${MAXJ:-9}
LOG=logs/alfworld_sweep.log
M=results/alfworld/${RUN_TAG:-qwen3-14b-main}
queue=()
for s in ${SEEDS:-0 1 2}; do for i in 0 1 2 3 4 5 6 7 8; do queue+=("$i $s"); done; done
declare -A started
victim() { python3 -c "import json;print(json.load(open('memloop/experiments/alfworld_pairs.json'))[$1]['victim'])"; }
busy() { ps -u $USER -o args | grep "[a]lfworld_recovery.py attack --pair-idx $1 --seed $2 " > /dev/null; }
while :; do
  running=$(ps -u $USER -o args | grep "[a]lfworld_job.sh" | sort -u | wc -l)
  left=0
  for q in "${queue[@]}"; do
    [ -n "${started[$q]}" ] && continue
    left=1
    set -- $q; D=$M/$(victim $1)/seed$2
    [ -f $D/eval_attack_B1_B2_DP_recovered_contain_recovered_b3_counterfactual.json ] && { started[$q]=1; continue; }
    ps -u $USER -o args | grep "[a]lfworld_job.sh $1 $2$" > /dev/null && { started[$q]=1; continue; }
    [ ! -f $D/attack.pkl ] && busy $1 $2 && continue
    [ "$running" -ge "$MAXJ" ] && break
    nohup bash scripts/alfworld_job.sh $1 $2 > /dev/null 2>&1 < /dev/null &
    started[$q]=1; running=$((running + 1))
  done
  [ $left = 0 ] && break
  sleep 60
done
echo "[$(date +%T)] SWEEP_QUEUE_DONE (all runs started)" >> $LOG
