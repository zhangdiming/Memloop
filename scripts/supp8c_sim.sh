#!/bin/bash
P=~/Memloop; cd $P
LOG=logs/supp8_sim.log
note() { echo "[$(date +%T)] $*" >> $LOG; }
XA=$1
while pgrep -u $USER -f supp8_fallback_job.sh > /dev/null; do sleep 30; done
note "supp8c: fallback done; second runner on 3005 3006"
{ for s in 0 1 2; do for i in 0 1 2 3 4 5 6 7 8; do echo "$i $s"; done; done; } \
  | PORTS="3005 3006" xargs -P 2 -L 1 bash scripts/supp8_sim_job.sh
while kill -0 $XA 2>/dev/null; do sleep 30; done
for pid in $(ps -u $USER -o pid,args | grep "[v]llm serve" | awk "{print \$1}"); do kill $pid; done
for pid in $(ps -u $USER -o pid,args | grep "[w]eb_agent_site.app" | awk "{print \$1}"); do kill $pid; done
note "SUPP8_SIM_DONE (servers stopped)"
