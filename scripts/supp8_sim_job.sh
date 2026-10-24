#!/bin/bash
IDX=$1; SEED=$2
P=~/Memloop; cd $P
export RUN_TAG=qwen3-14b-main
LOG=logs/supp8_sim.log
ARGS="--propagate --inject-num 5 --num-benign 20 --test-num 15 --eval-num 20"
TAGS="DFs_r075_f05_d0 DFs_r050_f05_d0 DFs_r025_f05_d0"
TAG=$(python3 -c "import json;print(json.load(open(\"baselines/MINJA/rap/victim_target_pair/victim_target.json\"))[$IDX][\"victim\"].replace(\" \",\"_\"))")
D=results/minja_recovery/$RUN_TAG/$TAG/seed$SEED
[ -f $D/eval_${TAGS// /_}.json ] && exit 0
mkdir $D/.sim_claim 2>/dev/null || exit 0
PORT=
for p in ${PORTS:-3000 3001 3003}; do
  exec {fd}>/tmp/webshop_port_$p.lock
  if flock -n $fd; then PORT=$p; break; fi
  exec {fd}>&-
done
[ -n "$PORT" ] || { echo "[$(date +%T)] $TAG/$SEED: no free port" >> $LOG; exit 1; }
export WEBSHOP_URL=http://localhost:$PORT/
echo "[$(date +%T)] $TAG/$SEED (:$PORT) start" >> $LOG
WORK_SUFFIX=_sim scripts/run_minja_recovery.sh $IDX $SEED eval --eval-on $TAGS $ARGS
[ -f $D/eval_${TAGS// /_}.json ] && R=ok || R=FAILED
echo "[$(date +%T)] $TAG/$SEED (:$PORT) done $R" >> $LOG
