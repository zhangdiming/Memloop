#!/bin/bash
P=~/Memloop; cd $P
LOG=logs/restart_determinism.log
ATT=$P/results/minja_recovery/qwen3-14b-det/cookies/seed0/attack.pkl
until grep -q TIME_SWEEP_DONE logs/time_sweep.log 2>/dev/null; do sleep 120; done
stop_vllm() {
  for pid in $(ps -eo pid,args | grep "[v]llm serve" | awk '{print $1}'); do kill $pid; done
  for i in $(seq 1 30); do ps -eo args | grep -q "[v]llm serve" || break; sleep 2; done
  sleep 10
}
start_vllm() {
  ( source ~/miniconda3/bin/activate memsec
    export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_BATCH_INVARIANT=1
    exec vllm serve Qwen/Qwen3-14B --served-model-name qwen3-14b --host 127.0.0.1 --port 8000 \
      --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.88 \
      --no-enable-prefix-caching --seed 0 \
      --default-chat-template-kwargs '{"enable_thinking": false}' "$@" ) > logs/vllm_probe.log 2>&1 < /dev/null &
  until curl -s localhost:8000/v1/models > /dev/null; do
    grep -q "Engine core initialization failed" logs/vllm_probe.log && { echo "vLLM FAILED ($*)" >> $LOG; exit 1; }
    sleep 10
  done
}
mkdir -p results/restart_determinism
echo "[$(date +%T)] start" >> $LOG
stop_vllm
for cfg in bi eager; do
  extra=; [ $cfg = eager ] && extra=--enforce-eager
  for r in 1 2; do
    start_vllm $extra
    ~/miniconda3/envs/minja-rap/bin/python scripts/restart_probe.py results/restart_determinism/${cfg}_run$r.json $ATT >> $LOG 2>&1
    echo "[$(date +%T)] $cfg run $r probed" >> $LOG
    stop_vllm
  done
  ~/miniconda3/envs/minja-rap/bin/python - results/restart_determinism/${cfg}_run1.json results/restart_determinism/${cfg}_run2.json >> $LOG <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
n = len(a["alone"])
print(f"  across restart: alone identical {sum(x == y for x, y in zip(a['alone'], b['alone']))}/{n}, "
      f"together identical {sum(x == y for x, y in zip(a['together'], b['together']))}/{n}")
PY
done
echo "[$(date +%T)] RESTART_DETERMINISM_DONE (vLLM left stopped)" >> $LOG
