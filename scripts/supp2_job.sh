#!/bin/bash
KIND=$1; IDX=$2; SEED=$3
P=~/Memloop; cd $P
export RUN_TAG=qwen3-14b-main
LOG=logs/supp2.log
ARGS="--propagate --inject-num 5 --num-benign 20 --test-num 15 --eval-num 20"
M=results/minja_recovery/$RUN_TAG
victim() { python3 -c "import json;print(json.load(open('baselines/MINJA/rap/victim_target_pair/victim_target.json'))[$1]['victim'].replace(' ','_'))"; }
TAG=$(victim $IDX)
if [ $KIND = ent ] && [ -f $M/$TAG/seed$SEED/recovered_contain_entry.pkl ]; then exit 0; fi
PORT=
for p in 3000 3001 3002 3003 3004; do
  exec {fd}>/tmp/webshop_port_$p.lock
  if flock -n $fd; then PORT=$p; break; fi
  exec {fd}>&-
done
[ -n "$PORT" ] || { echo "[$(date +%T)] $KIND $IDX/$SEED: no free port" >> $LOG; exit 1; }
export WEBSHOP_URL=http://localhost:$PORT/
run() { scripts/run_minja_recovery.sh $IDX $SEED "$@" $ARGS; }
note() { echo "[$(date +%T)] $KIND $IDX/$SEED (:$PORT) $*" >> $LOG; }
note start
case $KIND in
  ent) WORK_SUFFIX=_ent run recover --variant contain --seeding entry &&
       WORK_SUFFIX=_ent run eval --eval-on recovered_contain_entry ;;
  k2u) WORK_SUFFIX=_k2u run recover --variant contain --seeding union --accounts 2 &&
       WORK_SUFFIX=_k2u run eval --eval-on recovered_contain_union_k2 ;;
esac
note "done (exit $?)"
