#!/bin/bash
P=~/Memloop; cd $P
LOG=logs/listing_sweep.log
until grep -q SCALE_SWEEP_DONE logs/scale_sweep.log 2>/dev/null; do sleep 120; done
echo "[$(date +%T)] start; bringing up WebShop :3000-3004" >> $LOG
for port in 3000 3001 3002 3003 3004; do
  ss -ltn | grep -q ":$port " || scripts/webshop_instance.sh $port >> $LOG 2>&1
done
stop_vllm() {
  for pid in $(ps -eo pid,args | grep "[v]llm serve" | awk '{print $1}'); do kill $pid; done
  for i in $(seq 1 30); do ps -eo args | grep -q "[v]llm serve" || break; sleep 2; done
  sleep 10
}
start_vllm() {
  GPU_UTIL=0.72 nohup serve/$1 > logs/vllm_listing.log 2>&1 < /dev/null &
  until curl -s localhost:8000/v1/models > /dev/null; do
    grep -q "Engine core initialization failed" logs/vllm_listing.log && { echo "vLLM FAILED ($1)" >> $LOG; exit 1; }
    sleep 10
  done
  echo "[$(date +%T)] vLLM up: $1" >> $LOG
}
for spec in "qwen3-32b vllm_qwen3_32b_awq.sh" "qwen3-14b vllm_qwen3_14b.sh" "qwen3-8b vllm_qwen3_8b.sh"; do
  set -- $spec; M=$1; S=$2
  stop_vllm; start_vllm $S
  for i in 8 5 2; do for s in 0 1; do echo "$i $s $M"; done; done | \
    xargs -P 5 -L 1 bash -c 'scripts/listing_job.sh $0 $1 $2'
  echo "[$(date +%T)] $M done" >> $LOG
done
stop_vllm
for pid in $(ps -eo pid,args | grep "[w]eb_agent_site.app" | awk '{print $1}'); do kill $pid; done
echo "[$(date +%T)] LISTING_SWEEP_DONE (vLLM and WebShop stopped)" >> $LOG
