#!/bin/bash
IDX=$1; SEED=$2
P=~/Memloop; cd $P
export RUN_TAG=qwen3-14b-main
LOG=logs/supp7_df.log
ARGS="--propagate --inject-num 5 --num-benign 20 --test-num 15 --eval-num 20"
TAG=$(python3 -c "import json;print(json.load(open(\"baselines/MINJA/rap/victim_target_pair/victim_target.json\"))[$IDX][\"victim\"].replace(\" \",\"_\"))")
D=results/minja_recovery/$RUN_TAG/$TAG/seed$SEED
[ -f $D/DF.pkl ] && [ -f $D/DP.pkl ] || { echo "[$(date +%T)] $TAG/$SEED: no DF/DP state" >> $LOG; exit 0; }
[ -f $D/eval_DF_DP.json ] && exit 0
PORT=
for p in 3000 3001 3002 3003 3004; do
  exec {fd}>/tmp/webshop_port_$p.lock
  if flock -n $fd; then PORT=$p; break; fi
  exec {fd}>&-
done
[ -n "$PORT" ] || { echo "[$(date +%T)] $TAG/$SEED: no free port" >> $LOG; exit 1; }
export WEBSHOP_URL=http://localhost:$PORT/
echo "[$(date +%T)] $TAG/$SEED (:$PORT) start" >> $LOG
WORK_SUFFIX=_df scripts/run_minja_recovery.sh $IDX $SEED eval --eval-on DF DP $ARGS
echo "[$(date +%T)] $TAG/$SEED (:$PORT) done (exit $?)" >> $LOG
