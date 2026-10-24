#!/bin/bash
P=~/Memloop; cd $P
export RUN_TAG=qwen3-14b-disp-det MODEL=qwen3-14b
LOG=logs/disp_det.log
ARGS="--propagate --inject-num 5 --benign-attempts 40 --test-num 15 --eval-num 20 --attack-window 0.8 1.0 --num-retrieval 1"
for pid in $(ps -eo pid,args | grep "[v]llm serve" | awk '{print $1}'); do kill $pid; done; sleep 15
nohup serve/vllm_qwen3_14b_bi.sh > logs/vllm_disp_det.log 2>&1 < /dev/null &
until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done
echo "[$(date +%T)] batch-invariant vLLM up" >> $LOG
pair() {
  local IDX=$1; export WEBSHOP_URL=http://localhost:$2/
  r() { scripts/run_minja_recovery.sh $IDX 0 "$@" $ARGS; }
  r attack || { echo "[$(date +%T)] pair $IDX attack FAILED" >> $LOG; return; }
  echo "[$(date +%T)] pair $IDX attack done" >> $LOG
  r recover --variant full & a=$!
  r recover --variant b3 & b=$!
  r counterfactual & c=$!
  wait $a $b $c
  r compare
  echo "[$(date +%T)] pair $IDX done" >> $LOG
}
pair 8 3000 & pair 5 3001 & pair 2 3002 &
wait
for pid in $(ps -eo pid,args | grep "[v]llm serve" | awk '{print $1}'); do kill $pid; done
echo "[$(date +%T)] DISP_DET_DONE (vLLM stopped; WebShop still up)" >> $LOG
