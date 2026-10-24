#!/bin/bash
P=~/Memloop; cd $P
LOG=logs/pilot_recovery.log
echo "[$(date +%T)] starting vLLM" >> $LOG
nohup serve/vllm_qwen3_14b.sh > logs/vllm_qwen3_14b.log 2>&1 < /dev/null &
until curl -s localhost:8000/v1/models > /dev/null; do
  grep -q "Engine core initialization failed" logs/vllm_qwen3_14b.log && { echo "vLLM FAILED" >> $LOG; exit 1; }
  sleep 10
done
echo "[$(date +%T)] vLLM ready" >> $LOG
for port in 3000 3001 3002; do scripts/webshop_instance.sh $port >> $LOG 2>&1; done

ARGS="--propagate --inject-num 5 --num-benign 20 --test-num 15 --eval-num 10"
pair() {
  local IDX=$1 PORT=$2; export WEBSHOP_URL=http://localhost:$PORT/
  scripts/run_minja_recovery.sh $IDX 0 attack $ARGS || { echo "[$(date +%T)] pair $IDX attack FAILED" >> $LOG; return; }
  echo "[$(date +%T)] pair $IDX attack done" >> $LOG
  scripts/run_minja_recovery.sh $IDX 0 recover $ARGS & a=$!
  scripts/run_minja_recovery.sh $IDX 0 counterfactual $ARGS & b=$!
  scripts/run_minja_recovery.sh $IDX 0 baselines $ARGS & c=$!
  wait $a $b $c
  echo "[$(date +%T)] pair $IDX recover/counterfactual/baselines done" >> $LOG
  scripts/run_minja_recovery.sh $IDX 0 eval $ARGS
  echo "[$(date +%T)] pair $IDX eval done" >> $LOG
}
pair 8 3000 &
pair 5 3001 &
pair 2 3002 &
wait
echo "[$(date +%T)] PILOT_DONE" >> $LOG
