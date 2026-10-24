#!/bin/bash
P=~/Memloop; cd $P
LOG=logs/supp8_sim.log
note() { echo "[$(date +%T)] $*" >> $LOG; }
note "supp8b: waiting for running evals ($*)"
for pid in "$@"; do while kill -0 $pid 2>/dev/null; do sleep 20; done; done
kill $(cat logs/webshop_server_3004.pid) 2>/dev/null
note "supp8b: :3004 stopped; remaining evals on 3000 3001 3003"
{ for s in 0 1 2; do for i in 0 1 2 3 4 5 6 7 8; do echo "$i $s"; done; done; } \
  | PORTS="3000 3001 3003" xargs -P 3 -L 1 bash scripts/supp8_sim_job.sh
while pgrep -u $USER -f supp8_fallback_job.sh > /dev/null; do sleep 30; done
for pid in $(ps -u $USER -o pid,args | grep "[v]llm serve" | awk "{print \$1}"); do kill $pid; done
for pid in $(ps -u $USER -o pid,args | grep "[w]eb_agent_site.app" | awk "{print \$1}"); do kill $pid; done
note "SUPP8_SIM_DONE (servers stopped)"
