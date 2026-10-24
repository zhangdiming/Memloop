#!/bin/bash
P=~/Memloop; cd $P/memloop
O=$P/results/flip_rate/long; mkdir -p $O
LOG=$P/logs/flip_long.log
PY=~/miniconda3/envs/memsec/bin/python
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=EMPTY
stop_vllm() {
  for pid in $(ps -eo pid,args | grep "[v]llm serve" | awk '{print $1}'); do kill $pid; done
  for i in $(seq 1 30); do ps -eo args | grep -q "[v]llm serve" || break; sleep 2; done
  sleep 10
}
start_vllm() {
  nohup $P/serve/vllm_qwen3_14b_bi.sh > $P/logs/vllm_flip.log 2>&1 < /dev/null &
  until curl -s localhost:8000/v1/models > /dev/null; do
    grep -q "Engine core initialization failed" $P/logs/vllm_flip.log && { echo "vLLM FAILED" >> $LOG; exit 1; }
    sleep 10
  done
  grep -o "max_num_batched_tokens[^,]*\|chunked_prefill[^,]*" $P/logs/vllm_flip.log | sort -u | head -4 >> $LOG
}
run() { $PY experiments/flip_rate.py run --prompts $O/prompts.jsonl --out $O/$1.json --concurrency $2 --max-tokens 300 >> $LOG 2>&1; echo "[$(date +%T)] $1 done" >> $LOG; }
echo "[$(date +%T)] start" >> $LOG
$PY experiments/flip_rate.py prompts --n 200 --shots 60 --out $O/prompts.jsonl >> $LOG 2>&1
start_vllm; run bi_a01 1; run bi_a16 16; stop_vllm
start_vllm; run bi_b16 16; stop_vllm
$PY experiments/flip_rate.py compare $O/bi_a01.json $O/bi_a16.json $O/bi_b16.json --out $O/compare.json >> $LOG 2>&1
echo "[$(date +%T)] FLIP_LONG_DONE (vLLM stopped)" >> $LOG
