#!/bin/bash
P=~/Memloop; cd $P
LOG=logs/supp3_c005.log
note() { echo "[$(date +%T)] $*" >> $LOG; }
until grep -q SUPP3_C003_DONE logs/supp3_c003.log 2>/dev/null; do sleep 60; done
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=EMPTY HF_HUB_OFFLINE=1
stop_vllm() { for pid in $(ps -u $USER -o pid,args | grep "[v]llm serve" | awk '{print $1}'); do kill $pid; done; sleep 30; }
webshop() { for p in "$@"; do ss -ltn | grep -q ":$p " || bash scripts/webshop_instance.sh $p >> $LOG; done; }
stop_vllm
nohup serve/vllm_qwen3_14b.sh > logs/vllm_supp3_default.log 2>&1 < /dev/null &
VLLM=$!
until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done
webshop 3000 3001 3002 3003 3004
note "default vLLM up, WebShop 3000-3004 ready"
PREV=recovered_contain_entry
for R in 2 3; do
  ~/miniconda3/envs/memsec/bin/python memloop/experiments/detect_llm.py --state $PREV detector_flags_r$R.json >> $LOG 2>&1 \
    || { note "r$R detection FAILED"; break; }
  note "r$R detection done"
  { for s in 0 1 2; do for i in 0 1 2 3 4 5 6 7 8; do echo "$R $i $s $PREV"; done; done; } \
    | xargs -P 5 -L 1 bash scripts/supp3_c005_job.sh
  note "r$R recovery round done"
  PREV=recovered_contain_entry_r$R
done
kill $VLLM; stop_vllm
for pid in $(ps -u $USER -o pid,args | grep "[w]eb_agent_site.app" | awk '{print $1}'); do kill $pid; done
note "SUPP3_C005_DONE (servers stopped)"
