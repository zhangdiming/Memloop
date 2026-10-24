#!/bin/bash
P=~/Memloop; cd $P
for s in 0 1 2; do for i in 0 1 2 3 4 5 6 7 8; do echo "$i $s"; done; done | \
  xargs -P 5 -L 1 bash -c 'scripts/main_job.sh $0 $1'
echo "[$(date +%T)] MAIN_SWEEP_DONE" >> logs/main_sweep.log
