#!/bin/bash
PORT=${1:?port}
D=~/Memloop/logs
source ~/miniconda3/bin/activate webshop
cd ~/Memloop/baselines/MINJA/rap/webshop
FLASK_ENV= WEBSHOP_PORT=$PORT python -u -m web_agent_site.app --attrs > $D/webshop_server_$PORT.log 2>&1 &
echo $! > $D/webshop_server_$PORT.pid
until grep -q "Running on" $D/webshop_server_$PORT.log; do sleep 5; done
curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" localhost:$PORT/fixed_0_0 > $D/webshop_warmup_$PORT.status
echo "[$(date +%T)] port $PORT warm: $(cat $D/webshop_warmup_$PORT.status)"
