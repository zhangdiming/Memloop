#!/bin/bash
P=~/Memloop; cd $P
LOG=logs/main_sweep.log
echo "[$(date +%T)] switching vLLM to normal mode (gpu util 0.72)" >> $LOG
for pid in $(ps -eo pid,args | grep "[v]llm serve" | awk '{print $1}'); do kill $pid; done
sleep 20
GPU_UTIL=0.72 nohup serve/vllm_qwen3_14b.sh > logs/vllm_qwen3_14b.log 2>&1 < /dev/null &
until curl -s localhost:8000/v1/models > /dev/null; do
  grep -q "Engine core initialization failed" logs/vllm_qwen3_14b.log && { echo "vLLM FAILED" >> $LOG; exit 1; }
  sleep 10
done
echo "[$(date +%T)] vLLM ready" >> $LOG
for port in 3003 3004; do
  ss -ltn | grep -q ":$port " || scripts/webshop_instance.sh $port >> $LOG 2>&1
done
for port in 3000 3001 3002 3003 3004; do
  ss -ltn | grep -q ":$port " || { echo "WebShop :$port not listening" >> $LOG; exit 1; }
done
echo "[$(date +%T)] servers ready; starting sweep" >> $LOG
scripts/main_sweep.sh
