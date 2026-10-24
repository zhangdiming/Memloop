#!/bin/bash
P=~/Memloop; cd $P
PLAN="0 camera 3000
1 chair 3000
2 cookies 3001
3 curtain 3001
4 hair_extension 3002
5 jeans 3002
6 shampoo 3003
7 speaker 3004"
for pid in $(ps -eo pid,args | grep "[s]cripts/run_minja.sh [0-7] 0" | awk '{print $1}'); do kill $pid; done
for pid in $(ps -eo pid,args | grep "[m]inja.py --model_name" | awk '{print $1}'); do kill $pid; done
sleep 5
echo "$PLAN" | while read IDX TAG PORT; do
  R=results/minja_rap/qwen3-14b/$TAG/seed0
  mkdir -p $R/aborted_cpu_embed
  mv $R/*.json $R/run_meta.txt $R/aborted_cpu_embed/ 2>/dev/null
  [ -f logs/minja_pair${IDX}_seed0.log ] && mv logs/minja_pair${IDX}_seed0.log $R/aborted_cpu_embed/driver.log
  W=workdirs/minja_rap/qwen3-14b/$TAG/seed0/log
  mkdir -p $R/aborted_cpu_embed/logs && mv $W/*.log $R/aborted_cpu_embed/logs/ 2>/dev/null
  WEBSHOP_URL=http://localhost:$PORT/ nohup bash -c "scripts/run_minja.sh $IDX 0 qwen3-14b; echo MINJA_DONE" > logs/minja_pair${IDX}_seed0.log 2>&1 < /dev/null &
  echo "started $TAG on :$PORT"
  sleep 3
done
