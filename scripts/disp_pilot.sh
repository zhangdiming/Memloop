#!/bin/bash
P=~/Memloop; cd $P
LOG=logs/disp_pilot.log
until grep -q CF2_LISTING_DONE logs/cf2_listing.log 2>/dev/null; do sleep 120; done
echo "[$(date +%T)] start; bringing up WebShop :3000-3004" >> $LOG
for port in 3000 3001 3002 3003 3004; do
  ss -ltn | grep -q ":$port " || scripts/webshop_instance.sh $port >> $LOG 2>&1
done
GPU_UTIL=0.72 nohup serve/vllm_qwen3_14b.sh > logs/vllm_disp.log 2>&1 < /dev/null &
until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done
echo "[$(date +%T)] vLLM up" >> $LOG
job() {
  IDX=$1; SEED=$2; K=$3; export MODEL=qwen3-14b RUN_TAG=qwen3-14b-disp-k$3
  PORT=
  for p in 3000 3001 3002 3003 3004; do
    exec {fd}>/tmp/webshop_port_$p.lock
    if flock -n $fd; then PORT=$p; break; fi
    exec {fd}>&-
  done
  [ -n "$PORT" ] || { echo "[$(date +%T)] $IDX/$SEED/k$K: no free port" >> $LOG; return 1; }
  export WEBSHOP_URL=http://localhost:$PORT/
  ARGS="--propagate --inject-num 5 --benign-attempts 40 --test-num 15 --eval-num 20 --attack-window 0.8 1.0 --num-retrieval $K"
  r() { scripts/run_minja_recovery.sh $IDX $SEED "$@" $ARGS; }
  TAG=$(python3 -c "import json;print(json.load(open('baselines/MINJA/rap/victim_target_pair/victim_target.json'))[$IDX]['victim'].replace(' ','_'))")
  OUT=results/minja_recovery/$RUN_TAG/$TAG/seed$SEED
  echo "[$(date +%T)] k$K $TAG/seed$SEED (:$PORT) start" >> $LOG
  [ -f $OUT/attack.pkl ] || r attack || { echo "[$(date +%T)] k$K $TAG attack FAILED" >> $LOG; return 1; }
  [ -f $OUT/recovered.pkl ] || r recover --variant full || echo "[$(date +%T)] k$K $TAG recover FAILED" >> $LOG
  python3 -c "import json;s=json.load(open('$OUT/recovered_stats.json'));print('k$K $TAG/seed$SEED', s['steps_replayed'], 'after_tau', s['steps_after_tau'], 'checks', s['displacement_checks'])" >> $LOG
}
export -f job; export LOG
for k in 1 3; do for i in 8 5 2; do echo "$i 0 $k"; done; done | xargs -P 5 -L 1 bash -c 'job $0 $1 $2'
for pid in $(ps -eo pid,args | grep "[v]llm serve" | awk '{print $1}'); do kill $pid; done
echo "[$(date +%T)] DISP_PILOT_DONE (vLLM stopped; WebShop left up for the deterministic run)" >> $LOG
