#!/bin/bash
IDX=$1; SEED=$2; MODE=$3
P=~/Memloop; cd $P
export RUN_TAG=qwen3-14b-main
LOG=logs/supp11_blamed.log
ARGS="--propagate --inject-num 5 --num-benign 20 --test-num 15 --eval-num 20 --other-num 10"
R=results/minja_recovery/$RUN_TAG
victims=$(python3 -c "import json;print(\" \".join(x[\"victim\"].replace(\" \",\"_\") for x in json.load(open(\"baselines/MINJA/rap/victim_target_pair/victim_target.json\"))))")
name() { echo $victims | cut -d" " -f$(($1 + 1)); }
TAG=$(name $IDX)
D=$R/$TAG/seed$SEED
STATES="DB DBs_r075_f05_d0 DBs_r050_f05_d0 DBs_r025_f05_d0"
case $MODE in
  eval) DONE=$D/eval_${STATES// /_}.json;;
  utility) STATES="DB counterfactual_db"; DONE=$D/utility_counterfactual_db.json
           [ -f $DONE ] || mkdir $D/.cfdb_claim 2>/dev/null || exit 0;;
esac
[ -f $DONE ] && exit 0
PORT=
for p in ${WEBSHOP_PORTS:-3000 3001 3002 3003 3004 3005}; do
  exec {fd}>/tmp/webshop_port_$p.lock
  if flock -n $fd; then PORT=$p; break; fi
  exec {fd}>&-
done
[ -n "$PORT" ] || { echo "[$(date +%T)] $TAG/$SEED $MODE: no free port" >> $LOG; exit 1; }
export WEBSHOP_URL=http://localhost:$PORT/
run() {
  echo "[$(date +%T)] $3/$2 $MODE (:$PORT) start" >> $LOG
  WORK_SUFFIX=_db scripts/run_minja_recovery.sh $1 $2 $MODE --eval-on $STATES $ARGS
  rc=$?
  [ -f $R/$3/seed$2/$(basename $DONE) ] && ok=ok || ok=FAILED
  echo "[$(date +%T)] $3/$2 $MODE (:$PORT) done $ok (exit $rc)" >> $LOG
}
run $IDX $SEED $TAG
[ $MODE = utility ] || exit 0
while :; do
  found=
  for i in 0 1 2 3 4 5 6 7 8; do for s in 0 1 2; do
    d=$R/$(name $i)/seed$s
    [ -f $d/utility_DB.json ] && [ ! -f $d/utility_counterfactual_db.json ] || continue
    mkdir $d/.cfdb_claim 2>/dev/null || continue
    mv $d/utility_DB.json $d/utility_DB_first.json
    echo "[$(date +%T)] $(name $i)/$s utility redo with counterfactual_db" >> $LOG
    run $i $s $(name $i); found=1
  done; done
  [ -n "$found" ] || break
done
