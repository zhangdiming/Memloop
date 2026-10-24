#!/bin/bash
P=~/Memloop; cd $P
export RUN_TAG=qwen3-14b-det
LOG=logs/alfworld_det.log
note() { echo "[$(date +%T)] $*" >> $LOG; }
for pid in $(ps -u $USER -o pid,args | grep "[v]llm serve" | awk '{print $1}'); do kill $pid; done
sleep 30
nohup serve/vllm_qwen3_14b_bi.sh > logs/vllm_alfworld_bi.log 2>&1 < /dev/null &
VLLM=$!
until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done
note "batch-invariant vLLM up (pid $VLLM)"
job() {
  IDX=$1; R() { scripts/alfworld_run.sh $IDX 0 "$@"; }
  V=$(python3 -c "import json;print(json.load(open('memloop/experiments/alfworld_pairs.json'))[$IDX]['victim'])")
  D=results/alfworld/$RUN_TAG/$V/seed0
  note "$V start"
  [ -f $D/attack.pkl ] || R attack || { note "$V attack failed"; return; }
  [ -f $D/counterfactual.pkl ] || R counterfactual & a=$!
  [ -f $D/recovered_contain.pkl ] || WORK_SUFFIX=_contain R recover --variant contain & b=$!
  [ -f $D/recovered_b3.pkl ] || WORK_SUFFIX=_b3 R recover --variant b3 & c=$!
  wait $a $b $c
  R compare && note "$V done" || note "$V compare failed"
}
pids=()
for i in 0 1 2 3 4 5 6 7 8; do job $i & pids+=($!); done
wait "${pids[@]}"
kill $VLLM; sleep 20
for pid in $(ps -u $USER -o pid,args | grep "[v]llm serve" | awk '{print $1}'); do kill $pid; done
note "ALFWORLD_DET_DONE (server stopped)"
