#!/bin/bash
P=~/Memloop; cd $P
LOG=logs/supp10_utility.log
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=EMPTY HF_HUB_OFFLINE=1
ss -ltn | grep -q ":3005 " && { echo "[$(date +%T)] extra: port 3005 busy" >> $LOG; exit 1; }
bash scripts/webshop_instance.sh 3005 >> $LOG
WS=$(cat logs/webshop_server_3005.pid)
for r in "$@"; do
  WEBSHOP_PORTS=3005 bash scripts/supp10_utility_job.sh ${r%:*} ${r#*:}
done
kill $WS
echo "[$(date +%T)] extra: $* finished, WebShop 3005 ($WS) stopped" >> $LOG
