#!/bin/bash
P=~/Memloop; cd $P/memloop
O=$P/results/flip_rate; mkdir -p $O
LOG=$P/logs/flip_rate.log
PY=~/miniconda3/envs/memsec/bin/python
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=EMPTY
stop_vllm() {
  for pid in $(ps -eo pid,args | grep "[v]llm serve" | awk '{print $1}'); do kill $pid; done
  for i in $(seq 1 30); do ps -eo args | grep -q "[v]llm serve" || break; sleep 2; done
  sleep 10
}
start_vllm() {
  nohup $P/serve/$1 > $P/logs/vllm_flip.log 2>&1 < /dev/null &
  until curl -s localhost:8000/v1/models > /dev/null; do
    grep -q "Engine core initialization failed" $P/logs/vllm_flip.log && { echo "vLLM FAILED ($1)" >> $LOG; exit 1; }
    sleep 10
  done
}
run() { $PY experiments/flip_rate.py run --prompts $O/prompts.jsonl --out $O/$1.json --concurrency $2 >> $LOG 2>&1; echo "[$(date +%T)] $1 done" >> $LOG; }
echo "[$(date +%T)] start" >> $LOG
[ -f $O/prompts.jsonl ] || $PY experiments/flip_rate.py prompts --out $O/prompts.jsonl >> $LOG 2>&1
stop_vllm
start_vllm vllm_qwen3_14b_bi.sh; run bi_a32 32; run bi_a08 8; stop_vllm
start_vllm vllm_qwen3_14b_bi.sh; run bi_b32 32; stop_vllm
start_vllm vllm_qwen3_14b.sh;    run nm_a32 32; run nm_a32b 32; stop_vllm
$PY experiments/flip_rate.py compare $O/bi_a32.json $O/bi_a08.json $O/bi_b32.json $O/nm_a32.json $O/nm_a32b.json \
  --out $O/compare.json >> $LOG 2>&1
echo "[$(date +%T)] FLIP_RATE_DONE (vLLM stopped)" >> $LOG
