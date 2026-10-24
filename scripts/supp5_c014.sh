#!/bin/bash
P=~/Memloop; cd $P
LOG=logs/supp5.log
note() { echo "[$(date +%T)] $*" >> $LOG; }
stop_vllm() { for pid in $(ps -u $USER -o pid,args | grep "[v]llm serve" | awk '{print $1}'); do kill $pid; done; sleep 30; }
webshop() { for p in "$@"; do ss -ltn | grep -q ":$p " || bash scripts/webshop_instance.sh $p >> $LOG; done; }

stop_vllm
nohup serve/vllm_qwen3_14b_bi.sh > logs/vllm_supp5_bi.log 2>&1 < /dev/null &
until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done
webshop 3000 3001 3002 3003 3004 3005
note "batch-invariant vLLM up, WebShop 3000-3005 ready"

job() {
  local IDX=$1 PL=$3; export WEBSHOP_URL=http://localhost:$2/ RUN_TAG=qwen3-14b-inter-$3
  local ARGS="--propagate --inject-num 5 --test-num 15 --eval-num 10"
  if [ $PL = late ]; then ARGS="$ARGS --benign-attempts 40 --attack-window 0.8 1.0"
  else ARGS="$ARGS --num-benign 20"; fi
  run() { scripts/run_minja_recovery.sh $IDX 0 "$@" $ARGS; }
  note "$PL $IDX start"
  run attack || { note "$PL $IDX attack FAILED"; return; }
  note "$PL $IDX attack done"
  WORK_SUFFIX=_cfd run counterfactual --cf-tag counterfactual_during6 --during 6 & a=$!
  WORK_SUFFIX=_ri run recover --variant contain --during 6 --arrive interleaved --arrive-every 5 & b=$!
  wait $a; note "$PL $IDX counterfactual exit $?"
  wait $b; note "$PL $IDX recover exit $?"
  run compare --base counterfactual_during6
  note "$PL $IDX done"
}
job 8 3000 shuf & j1=$!
job 5 3001 shuf & j2=$!
job 2 3002 shuf & j3=$!
job 8 3003 late & j4=$!
job 5 3004 late & j5=$!
job 2 3005 late & j6=$!
wait $j1 $j2 $j3 $j4 $j5 $j6
stop_vllm
for pid in $(ps -u $USER -o pid,args | grep "[w]eb_agent_site.app" | awk '{print $1}'); do kill $pid; done
note "SUPP5_DONE (servers stopped)"
