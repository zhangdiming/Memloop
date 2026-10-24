#!/bin/bash
IDX=$1; SEED=$2; shift 2
STATES=${*:-recovered_contain DF DP counterfactual B2 B1 empty}
P=~/Memloop; cd $P
export RUN_TAG=qwen3-14b-main
LOG=logs/supp10_utility.log
ARGS="--propagate --inject-num 5 --num-benign 20 --test-num 15 --eval-num 20 --other-num 10"
TAG=$(python3 -c "import json;print(json.load(open(\"baselines/MINJA/rap/victim_target_pair/victim_target.json\"))[$IDX][\"victim\"].replace(\" \",\"_\"))")
D=results/minja_recovery/$RUN_TAG/$TAG/seed$SEED
todo=$(for s in $STATES; do [ -f $D/utility_$s.json ] || echo $s; done)
[ -n "$todo" ] || exit 0
PORT=
for p in ${WEBSHOP_PORTS:-3000 3001 3002 3003 3004}; do
  exec {fd}>/tmp/webshop_port_$p.lock
  if flock -n $fd; then PORT=$p; break; fi
  exec {fd}>&-
done
[ -n "$PORT" ] || { echo "[$(date +%T)] $TAG/$SEED: no free port" >> $LOG; exit 1; }
export WEBSHOP_URL=http://localhost:$PORT/
echo "[$(date +%T)] $TAG/$SEED (:$PORT) start:" $todo >> $LOG
WORK_SUFFIX=_util scripts/run_minja_recovery.sh $IDX $SEED utility --eval-on $todo $ARGS
rc=$?
echo "[$(date +%T)] $TAG/$SEED (:$PORT) done (exit $rc)" >> $LOG
