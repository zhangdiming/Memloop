#!/bin/bash
P=~/Memloop; cd $P
LOG=logs/alfworld_sweep.log
nohup serve/vllm_qwen3_14b.sh > logs/vllm_alfworld_rep.log 2>&1 < /dev/null &
VLLM=$!
until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done
echo "[$(date +%T)] REP default vLLM up (pid $VLLM)" >> $LOG
SEEDS="3 4 5" MAXJ=${MAXJ:-18} bash scripts/alfworld_sweep.sh
while ps -u $USER -o args | grep -q "[a]lfworld_job.sh"; do sleep 60; done
kill $VLLM; sleep 20
for pid in $(ps -u $USER -o pid,args | grep "[v]llm serve" | awk '{print $1}'); do kill $pid; done
echo "[$(date +%T)] ALFWORLD_REP_DONE (server stopped)" >> $LOG
