#!/bin/bash
P=~/Memloop; cd $P
LOG=logs/parallel_det.log
until grep -q MAIN_SWEEP_DONE logs/main_sweep.log; do sleep 120; done
echo "[$(date +%T)] main sweep done; switching vLLM to batch-invariant mode" >> $LOG
for pid in $(ps -eo pid,args | grep "[v]llm serve" | awk '{print $1}'); do kill $pid; done
sleep 30
nohup serve/vllm_qwen3_14b_bi.sh > logs/vllm_qwen3_14b_bi.log 2>&1 < /dev/null &
until curl -s localhost:8000/v1/models > /dev/null; do
  grep -q "Engine core initialization failed" logs/vllm_qwen3_14b_bi.log && { echo "vLLM FAILED" >> $LOG; exit 1; }
  sleep 10
done
echo "[$(date +%T)] vLLM (batch-invariant) ready" >> $LOG
export RUN_TAG=qwen3-14b-det WEBSHOP_URL=http://localhost:3002/
ARGS="--propagate --inject-num 5 --num-benign 20 --test-num 15 --eval-num 10"
for w in 1 4 8; do
  WORK_SUFFIX=_par$w scripts/run_minja_recovery.sh 2 0 recover $ARGS --variant contain --parallel $w
  echo "[$(date +%T)] cookies contain parallel=$w done" >> $LOG
done
scripts/run_minja_recovery.sh 2 0 compare $ARGS
echo "[$(date +%T)] PARALLEL_DET_DONE" >> $LOG
