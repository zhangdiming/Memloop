#!/bin/bash
P=~/Memloop; cd $P
export RUN_TAG=qwen3-14b-main
LOG=logs/supp_default.log
until grep -q SUPP_DET_DONE logs/supp_det.log 2>/dev/null; do sleep 120; done
for pid in $(ps -u $USER -o pid,args | grep "[v]llm serve" | awk '{print $1}'); do kill $pid; done
sleep 30
nohup serve/vllm_qwen3_14b.sh > logs/vllm_supp_default.log 2>&1 < /dev/null &
until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done
for p in 3003 3004; do
  ss -ltn | grep -q ":$p " || bash scripts/webshop_instance.sh $p
done
echo "[$(date +%T)] default vLLM up, ports ready" >> $LOG
{
  for s in 0 1 2; do for i in 0 1 2 3 4 5 6 7 8; do echo "ck $i $s"; done; done
  while read i s; do echo "adv $i $s"; done < logs/union_k2_todo.txt
  for i in 0 1 2 3 4 5 6 7 8; do echo "all $i 0"; done
} | xargs -P 5 -L 1 bash scripts/supp_default_job.sh
echo "[$(date +%T)] SUPP_DEFAULT_DONE" >> $LOG
