#!/bin/bash
P=~/Memloop; cd $P
LOG=logs/alfworld_cf2.log
note() { echo "[$(date +%T)] $*" >> $LOG; }
nohup serve/vllm_qwen3_14b.sh > logs/vllm_alfworld_cf2.log 2>&1 < /dev/null &
VLLM=$!
until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done
note "default vLLM up (pid $VLLM)"
job() {
  IDX=$1; SEED=$2
  V=$(python3 -c "import json;print(json.load(open('memloop/experiments/alfworld_pairs.json'))[$IDX]['victim'])")
  D=results/alfworld/qwen3-14b-main/$V/seed$SEED
  [ -f $D/eval_counterfactual2.json ] && return
  [ -f $D/counterfactual2.pkl ] || WORK_SUFFIX=2 scripts/alfworld_run.sh $IDX $SEED counterfactual --cf-tag counterfactual2 \
    || { note "$V/$SEED cf2 failed"; return; }
  WORK_SUFFIX=_cf2 scripts/alfworld_run.sh $IDX $SEED eval --eval-on counterfactual2 && note "$V/$SEED done" || note "$V/$SEED eval failed"
}
export -f job note; export LOG
for s in 0 1 2; do for i in 0 1 2 3 4 5 6 7 8; do echo "$i $s"; done; done | xargs -P ${MAXJ:-18} -L 1 bash -c 'job $0 $1'
kill $VLLM; sleep 20
for pid in $(ps -u $USER -o pid,args | grep "[v]llm serve" | awk '{print $1}'); do kill $pid; done
note "ALFWORLD_CF2_DONE (server stopped)"
