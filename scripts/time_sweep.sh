#!/bin/bash
P=~/Memloop; cd $P
LOG=logs/time_sweep.log
until grep -q PARALLEL_DET_DONE logs/parallel_det.log 2>/dev/null; do sleep 120; done
echo "[$(date +%T)] parallel experiment done; switching vLLM to normal mode (gpu util 0.72)" >> $LOG
for pid in $(ps -eo pid,args | grep "[v]llm serve" | awk '{print $1}'); do kill $pid; done
sleep 30
GPU_UTIL=0.72 nohup serve/vllm_qwen3_14b.sh > logs/vllm_qwen3_14b.log 2>&1 < /dev/null &
until curl -s localhost:8000/v1/models > /dev/null; do
  grep -q "Engine core initialization failed" logs/vllm_qwen3_14b.log && { echo "vLLM FAILED" >> $LOG; exit 1; }
  sleep 10
done
for port in 3000 3001 3002 3003 3004; do
  ss -ltn | grep -q ":$port " || { echo "WebShop :$port not listening" >> $LOG; exit 1; }
done
echo "[$(date +%T)] servers ready; starting time sweep" >> $LOG
for w in early mid late; do for i in 5 4 2; do echo "$i 0 $w"; done; done | \
  xargs -P 5 -L 1 bash -c 'scripts/time_job.sh $0 $1 $2'
echo "[$(date +%T)] TIME_SWEEP_DONE" >> $LOG
