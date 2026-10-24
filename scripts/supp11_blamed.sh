#!/bin/bash
P=~/Memloop; cd $P
LOG=logs/supp11_blamed.log
note() { echo "[$(date +%T)] $*" >> $LOG; }
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=EMPTY HF_HUB_OFFLINE=1
PORTS="3000 3001 3002 3003 3004 3005"
if curl -s localhost:8000/v1/models > /dev/null; then note "port 8000 already serving: abort"; exit 1; fi
for p in $PORTS; do ss -ltn | grep -q ":$p " && { note "port $p busy: abort"; exit 1; }; done
nohup serve/vllm_qwen3_14b.sh > logs/vllm_supp11.log 2>&1 < /dev/null &
VLLM=$!
until curl -s localhost:8000/v1/models > /dev/null; do
  kill -0 $VLLM 2> /dev/null || { note "vLLM exited"; exit 1; }; sleep 10
done
WS=()
for p in $PORTS; do
  bash scripts/webshop_instance.sh $p >> $LOG
  WS+=($(cat logs/webshop_server_$p.pid))
done
note "vLLM $VLLM and WebShop ${WS[*]} ready; $(free -g | awk '/Mem:/{print "RAM used " $3 "G, available " $7 "G"}')"
{ for m in eval utility; do for s in 0 1 2; do for i in 0 1 2 3 4 5 6 7 8; do echo "$i $s $m"; done; done; done; } \
  | WEBSHOP_PORTS="$PORTS" xargs -P 6 -L 1 bash scripts/supp11_blamed_job.sh
kill $VLLM ${WS[*]}
note "SUPP11_BLAMED_DONE (servers stopped)"
