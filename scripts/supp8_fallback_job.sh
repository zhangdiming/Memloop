#!/bin/bash
T=$1
P=~/Memloop; cd $P
export RUN_TAG=qwen3-14b-main
LOG=logs/supp8_sim.log
ARGS="--propagate --inject-num 5 --num-benign 20 --test-num 15 --eval-num 20"
case $T in r075*) PORT=3005;; r050*) PORT=3006;; *) PORT=3007;; esac
export WEBSHOP_URL=http://localhost:$PORT/
D=results/minja_recovery/$RUN_TAG/camera/seed2
echo "[$(date +%T)] camera/2 fallback $T (:$PORT) start" >> $LOG
[ -f $D/recovered_contain_sim_$T.pkl ] || WORK_SUFFIX=_sim_$T scripts/run_minja_recovery.sh 0 2 recover $ARGS \
  --variant contain --seeding entry --flags detector_flags_sim_$T.json --out-tag recovered_contain_sim_$T
WORK_SUFFIX=_sim_$T scripts/run_minja_recovery.sh 0 2 eval $ARGS --eval-on recovered_contain_sim_$T
echo "[$(date +%T)] camera/2 fallback $T done (exit $?)" >> $LOG
