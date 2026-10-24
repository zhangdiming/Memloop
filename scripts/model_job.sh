#!/bin/bash
IDX=${1:?idx}; SEED=${2:?seed}; MODEL=${3:?model}
P=~/Memloop; cd $P
export MODEL RUN_TAG=$MODEL-scale
LOG=logs/scale_sweep.log
ARGS="--propagate --inject-num 5 --benign-attempts 40 --test-num 15 --eval-num 20"
PORT=
for p in 3000 3001 3002 3003 3004; do
  exec {fd}>/tmp/webshop_port_$p.lock
  if flock -n $fd; then PORT=$p; break; fi
  exec {fd}>&-
done
[ -n "$PORT" ] || { echo "[$(date +%T)] $MODEL $IDX/$SEED: no free port" >> $LOG; exit 1; }
export WEBSHOP_URL=http://localhost:$PORT/
TAG=$(python3 -c "import json;print(json.load(open('baselines/MINJA/rap/victim_target_pair/victim_target.json'))[$IDX]['victim'].replace(' ','_'))")
OUT=results/minja_recovery/$RUN_TAG/$TAG/seed$SEED
run() { scripts/run_minja_recovery.sh $IDX $SEED "$@" $ARGS; }
note() { echo "[$(date +%T)] $MODEL $TAG/seed$SEED (:$PORT) $*" >> $LOG; }
note start
[ -f $OUT/attack.pkl ] || { run attack || { note "attack FAILED"; exit 1; }; note "attack done"; }
pids=()
[ -f $OUT/recovered_contain.pkl ] || { run recover --variant contain & pids+=($!); }
[ -f $OUT/counterfactual.pkl ]    || { run counterfactual & pids+=($!); }
[ -f $OUT/B2.pkl ]                || { run baselines & pids+=($!); }
for p in "${pids[@]}"; do wait $p || note "a phase FAILED (pid $p)"; done
[ -f $OUT/eval_attack_B1_B2_recovered_contain_counterfactual.json ] || \
  run eval --eval-on attack B1 B2 recovered_contain counterfactual
run compare
note "DONE"
