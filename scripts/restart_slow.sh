#!/bin/bash
P=~/Memloop; cd $P
for pair in "3 curtain 3001" "4 hair_extension 3002" "6 shampoo 3003" "7 speaker 3004"; do
  set -- $pair; IDX=$1; TAG=$2; PORT=$3
  for pid in $(ps -eo pid,args | grep "[s]cripts/run_minja.sh $IDX 0" | awk '{print $1}'); do kill $pid; done
  for pid in $(ps -eo pid,args | grep "[m]inja.py" | grep -- "--victim ${TAG//_/ } " | awk '{print $1}'); do kill $pid; done
  sleep 2
  R=results/minja_rap/qwen3-14b/$TAG/seed0
  mkdir -p $R/aborted_shared_server && mv $R/*.json $R/aborted_shared_server/ 2>/dev/null
  mv logs/minja_pair${IDX}_seed0.log $R/aborted_shared_server/driver.log
  WEBSHOP_URL=http://localhost:$PORT/ MINJA_EMB_DEVICE=cpu nohup bash -c "scripts/run_minja.sh $IDX 0 qwen3-14b; echo MINJA_DONE" > logs/minja_pair${IDX}_seed0.log 2>&1 < /dev/null &
  echo "restarted $TAG on :$PORT"
done
