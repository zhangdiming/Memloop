#!/bin/bash
P=~/Memloop; cd $P
MODEL=${MODEL:-qwen3-14b}
LOG=logs/qa_sweep_${MODEL:-qwen3-14b}.log
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=EMPTY OMP_NUM_THREADS=2
PY=~/miniconda3/envs/memsec/bin/python
job() {
  local OUT=$P/results/qa_recovery/$MODEL/$1/seed$2
  mkdir -p $OUT
  local A="--subject $1 --seed $2 --model $MODEL --propagate --out $OUT"
  r() { $PY memloop/experiments/qa_recovery.py "$@" $A >> $OUT/${1}${CF:-}.log 2>&1; }
  echo "[$(date +%T)] $1/seed$2 start" >> $LOG
  [ -f $OUT/attack.pkl ] || r attack || { echo "[$(date +%T)] $1/seed$2 attack FAILED" >> $LOG; return; }
  echo "[$(date +%T)] $1/seed$2 attack done: $(grep -h 'attack done' $OUT/attack.log | tail -1 | cut -c12-)" >> $LOG
  [ -f $OUT/recovered_contain.pkl ] || r recover & a=$!
  [ -f $OUT/counterfactual.pkl ] || r counterfactual & b=$!
  [ -f $OUT/counterfactual2.pkl ] || CF=2 r counterfactual --cf-tag counterfactual2 & c=$!
  [ -f $OUT/B2.pkl ] || r baselines & d=$!
  wait $a $b $c $d
  r eval; r compare
  echo "[$(date +%T)] $1/seed$2 DONE" >> $LOG
}
for s in ${SUBJECTS:-nutrition moral_scenarios}; do for k in ${SEEDS:-0 1 2}; do job $s $k & done; done
wait
echo "[$(date +%T)] QA_SWEEP_DONE" >> $LOG
