#!/bin/bash
D=~/Memloop/logs
source ~/miniconda3/bin/activate webshop
cd ~/Memloop/baselines/MINJA/rap/webshop
FLASK_ENV= python -u -m web_agent_site.app --log --attrs > $D/webshop_server.log 2>&1 &
echo $! > $D/webshop_server.pid
until grep -q "Running on" $D/webshop_server.log; do sleep 5; done
echo "[$(date +%T)] server up, warming"
curl -s -o $D/webshop_warmup.html -w "%{http_code} %{time_total}s\n" localhost:3000/fixed_0_0 > $D/webshop_warmup.status
echo "[$(date +%T)] warmup: $(cat $D/webshop_warmup.status) rss=$(($(ps -o rss= -p $(cat $D/webshop_server.pid))/1048576))G"
grep -q "^200" $D/webshop_warmup.status || { echo "warmup failed"; exit 1; }
echo "[$(date +%T)] MINJA toothbrush start"
~/Memloop/scripts/run_minja.sh 8 0 qwen3-14b
echo "[$(date +%T)] MINJA_DONE"
