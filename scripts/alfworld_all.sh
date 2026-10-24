#!/bin/bash
P=~/Memloop; cd $P
LOG=logs/alfworld_all.log
note() { echo "[$(date +%T)] $*" >> $LOG; }
nohup serve/vllm_qwen3_14b.sh > logs/vllm_alfworld_all.log 2>&1 < /dev/null &
VLLM=$!
until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done
note "default vLLM up (pid $VLLM)"
job() {
  IDX=$1; SEED=$2
  V=$(python3 -c "import json;print(json.load(open('memloop/experiments/alfworld_pairs.json'))[$IDX]['victim'])")
  D=results/alfworld/qwen3-14b-main/$V/seed$SEED
  [ -f $D/eval_recovered_all.json ] && return
  [ -f $D/recovered_all.pkl ] || WORK_SUFFIX=_all scripts/alfworld_run.sh $IDX $SEED recover --variant all \
    || { note "$V/$SEED recover failed"; return; }
  WORK_SUFFIX=_all scripts/alfworld_run.sh $IDX $SEED eval --eval-on recovered_all && note "$V/$SEED done" || note "$V/$SEED eval failed"
}
export -f job note; export LOG
for s in 0 1 2; do for i in 0 1 2 3 4 5 6 7 8; do echo "$i $s"; done; done | xargs -P ${MAXJ:-18} -L 1 bash -c 'job $0 $1'
kill $VLLM; sleep 20
for pid in $(ps -u $USER -o pid,args | grep "[v]llm serve" | awk '{print $1}'); do kill $pid; done
note "ALFWORLD_ALL_DONE (server stopped)"
