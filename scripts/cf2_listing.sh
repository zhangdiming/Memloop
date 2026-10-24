#!/bin/bash
P=~/Memloop; cd $P
LOG=logs/cf2_listing.log
until grep -q LISTING_SWEEP_DONE logs/listing_sweep.log 2>/dev/null; do sleep 120; done
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
  GPU_UTIL=0.72 nohup serve/$1 > logs/vllm_cf2.log 2>&1 < /dev/null &
  until curl -s localhost:8000/v1/models > /dev/null; do
    grep -q "Engine core initialization failed" logs/vllm_cf2.log && { echo "vLLM FAILED ($1)" >> $LOG; exit 1; }
    sleep 10
  done
  echo "[$(date +%T)] vLLM up: $1" >> $LOG
}
job() {
  IDX=$1; SEED=$2; export MODEL=$3 RUN_TAG=$3-listing
  PORT=
  for p in 3000 3001 3002 3003 3004; do
    exec {fd}>/tmp/webshop_port_$p.lock
    if flock -n $fd; then PORT=$p; break; fi
    exec {fd}>&-
  done
  [ -n "$PORT" ] || { echo "[$(date +%T)] $MODEL $IDX/$SEED: no free port" >> $LOG; return 1; }
  export WEBSHOP_URL=http://localhost:$PORT/
  TAG=$(python3 -c "import json;print(json.load(open('baselines/MINJA/rap/victim_target_pair/victim_target.json'))[$IDX]['victim'].replace(' ','_'))")
  OUT=results/minja_recovery/$RUN_TAG/$TAG/seed$SEED
  [ -f $OUT/counterfactual2.pkl ] && return 0
  echo "[$(date +%T)] $MODEL $TAG/seed$SEED (:$PORT) cf2 start" >> $LOG
  WORK_SUFFIX=2 scripts/run_minja_recovery.sh $IDX $SEED counterfactual --cf-tag counterfactual2 \
    --attack listing --propagate --inject-num 5 --benign-attempts 40 --test-num 15 --eval-num 20 \
    && echo "[$(date +%T)] $MODEL $TAG/seed$SEED cf2 DONE" >> $LOG \
    || echo "[$(date +%T)] $MODEL $TAG/seed$SEED cf2 FAILED" >> $LOG
}
export -f job; export LOG
source ~/miniconda3/bin/activate memsec
for spec in "qwen3-8b vllm_qwen3_8b.sh" "qwen3-14b vllm_qwen3_14b.sh" "qwen3-32b vllm_qwen3_32b_awq.sh"; do
  set -- $spec; M=$1; S=$2
  stop_vllm; start_vllm $S
  for i in 8 5 2; do for s in 0 1; do echo "$i $s $M"; done; done | \
    xargs -P 5 -L 1 bash -c 'job $0 $1 $2'
  python scripts/distributional.py $P/results/minja_recovery/$M-listing > results/minja_recovery/$M-listing/distributional.txt 2>&1
  echo "[$(date +%T)] $M cf2 done" >> $LOG
done
stop_vllm
for pid in $(ps -eo pid,args | grep "[w]eb_agent_site.app" | awk '{print $1}'); do kill $pid; done
echo "[$(date +%T)] CF2_LISTING_DONE (vLLM and WebShop stopped)" >> $LOG
