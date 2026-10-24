#!/bin/bash
R=$1; IDX=$2; SEED=$3; PREV=$4; FLAGS=$5
P=~/Memloop; cd $P
export RUN_TAG=qwen3-14b-main
LOG=logs/supp4_c005w.log
ARGS="--propagate --inject-num 5 --num-benign 20 --test-num 15 --eval-num 20"
M=results/minja_recovery/$RUN_TAG
TAG=$(python3 -c "import json;print(json.load(open('baselines/MINJA/rap/victim_target_pair/victim_target.json'))[$IDX]['victim'].replace(' ','_'))")
D=$M/$TAG/seed$SEED
NEW=recovered_contain_writer_r$R
[ -f $D/$PREV.pkl ] || exit 0
[ -f $D/$NEW.pkl ] && [ -f $D/eval_$NEW.json ] && exit 0
NF=$(python3 -c "import json;print(sum(v['flag'] for v in json.load(open('$D/$FLAGS')).values()))")
if [ "$NF" = 0 ]; then echo "[$(date +%T)] r$R $IDX/$SEED: detector quiet, stop" >> $LOG; exit 0; fi
PORT=
for p in 3000 3001 3002 3003 3004; do
  exec {fd}>/tmp/webshop_port_$p.lock
  if flock -n $fd; then PORT=$p; break; fi
  exec {fd}>&-
done
[ -n "$PORT" ] || { echo "[$(date +%T)] r$R $IDX/$SEED: no free port" >> $LOG; exit 1; }
export WEBSHOP_URL=http://localhost:$PORT/
run() { scripts/run_minja_recovery.sh $IDX $SEED "$@" $ARGS; }
echo "[$(date +%T)] r$R $IDX/$SEED (:$PORT) start, $NF flagged" >> $LOG
WORK_SUFFIX=_w$R run recover --variant contain --seeding writer --accounts 0 --from-state $PREV \
    --flags $FLAGS --out-tag $NEW &&
WORK_SUFFIX=_w$R run eval --eval-on $NEW
echo "[$(date +%T)] r$R $IDX/$SEED (:$PORT) done (exit $?)" >> $LOG
