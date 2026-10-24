#!/bin/bash
KIND=$1; IDX=$2; SEED=$3
P=~/Memloop; cd $P
export RUN_TAG=qwen3-14b-main
LOG=logs/supp_default.log
ARGS="--propagate --inject-num 5 --num-benign 20 --test-num 15 --eval-num 20"
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
  ck)  WORK_SUFFIX=_ck run baselines --baselines CK && WORK_SUFFIX=_ck run eval --eval-on CK ;;
  adv) WORK_SUFFIX=_k2o run recover --variant contain --seeding origin --accounts 2 & a=$!
       WORK_SUFFIX=_k2u run recover --variant contain --seeding union --accounts 2 & b=$!
       wait $a $b
       WORK_SUFFIX=_k2 run eval --eval-on recovered_contain_origin_k2 recovered_contain_union_k2 ;;
  all) WORK_SUFFIX=_all run recover --variant all &&
       WORK_SUFFIX=_all run eval --eval-on recovered_all &&
       WORK_SUFFIX=_all run compare ;;
esac
note "done (exit $?)"
