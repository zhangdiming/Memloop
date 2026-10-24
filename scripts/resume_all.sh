#!/bin/bash
P=~/Memloop; cd $P
echo "[$(date +%T)] starting vLLM"
nohup serve/vllm_qwen3_14b.sh > logs/vllm_qwen3_14b.log 2>&1 < /dev/null &
until curl -s localhost:8000/v1/models > /dev/null; do
  grep -q "Engine core initialization failed" logs/vllm_qwen3_14b.log && { echo "vLLM FAILED"; exit 1; }
  sleep 10
done
echo "[$(date +%T)] vLLM ready"
for port in 3000 3001 3002 3003 3004; do scripts/webshop_instance.sh $port; done
for port in 3000 3001 3002 3003 3004; do
  grep -q "^200" logs/webshop_warmup_$port.status || { echo "WebShop :$port FAILED"; exit 1; }
done
echo "[$(date +%T)] all WebShop instances warm"
PLAN="0 camera 3000
1 chair 3000
2 cookies 3001
3 curtain 3001
4 hair_extension 3002
5 jeans 3002
6 shampoo 3003
7 speaker 3004"
echo "$PLAN" | while read IDX TAG PORT; do
  R=results/minja_rap/qwen3-14b/$TAG/seed0
  W=workdirs/minja_rap/qwen3-14b/$TAG/seed0/log
  mkdir -p $R/aborted_paused && mv $W/*.log $R/aborted_paused/ 2>/dev/null
  rm -f $R/memory.json $R/test_memory.json
  WEBSHOP_URL=http://localhost:$PORT/ nohup bash -c "scripts/run_minja.sh $IDX 0 qwen3-14b; echo MINJA_DONE" > logs/minja_pair${IDX}_seed0.log 2>&1 < /dev/null &
  echo "[$(date +%T)] started $TAG on :$PORT"
  sleep 3
done
echo "[$(date +%T)] RESUME_DONE"
