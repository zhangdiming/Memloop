#!/bin/bash
P=~/Memloop; cd $P
LOG=logs/supp7_df.log
note() { echo "[$(date +%T)] $*" >> $LOG; }
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=EMPTY HF_HUB_OFFLINE=1
stop_vllm() { for pid in $(ps -u $USER -o pid,args | grep "[v]llm serve" | awk "{print \$1}"); do kill $pid; done; sleep 30; }
webshop() { for p in "$@"; do ss -ltn | grep -q ":$p " || bash scripts/webshop_instance.sh $p >> $LOG; done; }
stop_vllm
nohup serve/vllm_qwen3_14b.sh > logs/vllm_supp7_default.log 2>&1 < /dev/null &
VLLM=$!
until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done
webshop 3000 3001 3002 3003 3004
note "default vLLM up, WebShop 3000-3004 ready"
{ for s in 0 1 2; do for i in 0 1 2 3 4 5 6 7 8; do echo "$i $s"; done; done; } \
  | xargs -P 5 -L 1 bash scripts/supp7_df_job.sh
kill $VLLM; stop_vllm
for pid in $(ps -u $USER -o pid,args | grep "[w]eb_agent_site.app" | awk "{print \$1}"); do kill $pid; done
note "SUPP7_DF_DONE (servers stopped)"
