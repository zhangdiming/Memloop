#!/bin/bash
P=~/Memloop; cd $P
export RUN_TAG=qwen3-14b-main
LOG=logs/detector_recovery.log
ARGS="--propagate --inject-num 5 --num-benign 20 --test-num 15 --eval-num 20"
job() {
  export WEBSHOP_URL=http://localhost:$3/
  WORK_SUFFIX=_entry scripts/run_minja_recovery.sh $1 $2 recover $ARGS --variant contain --seeding entry \
    && echo "[$(date +%T)] $1/$2 entry recovery done" >> $LOG
  WORK_SUFFIX=_entry scripts/run_minja_recovery.sh $1 $2 eval $ARGS --eval-on recovered_contain_entry \
    && echo "[$(date +%T)] $1/$2 entry eval done" >> $LOG
}
job 8 0 3000 & job 8 1 3001 & job 2 0 3002 & job 3 1 3003 & job 3 2 3004 &
wait
echo "[$(date +%T)] DETECTOR_RECOVERY_DONE" >> $LOG
