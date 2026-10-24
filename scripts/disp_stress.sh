#!/bin/bash
P=~/Memloop; cd $P
LOG=logs/disp_stress.log
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=EMPTY OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=~/miniconda3/envs/memsec/bin/python
for pid in $(ps -eo pid,args | grep "[v]llm serve" | awk '{print $1}'); do kill $pid; done; sleep 10
nohup serve/vllm_qwen3_14b_bi.sh > logs/vllm_disp_stress.log 2>&1 < /dev/null &
VLLM=$!
until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done
echo "[$(date +%T)] batch-invariant vLLM up" >> $LOG
job() {
  local W; [ $4 = early ] && W="0.05 0.45" || W="0.55 0.95"
  local OUT=$P/results/disp_stress/$1/k$2/ov$3/$4/seed$5
  mkdir -p $OUT
  local A="--subject nutrition --seed $5 --model qwen3-14b --propagate --num-templates 5 --num-benign 60
           --max-benign-retries 3 --num-test 10 --retriever $1 --n-shots $2 --overlap $3 --attack-window $W --out $OUT"
  r() { $PY memloop/experiments/qa_recovery.py "$@" $A >> $OUT/${1}${V:-}.log 2>&1; }
  [ -f $OUT/attack.pkl ] || r attack || { echo "[$(date +%T)] $OUT attack FAILED" >> $LOG; return; }
  [ -f $OUT/recovered_contain.pkl ] || r recover & a=$!
  [ -f $OUT/recovered_contain_b3.pkl ] || V=_b3 r recover --variant b3 & b=$!
  [ -f $OUT/counterfactual.pkl ] || r counterfactual & c=$!
  wait $a $b $c
  r compare
  echo "[$(date +%T)] $1/k$2/ov$3/$4/seed$5 done" >> $LOG
}
for s in 0 1; do for w in early late; do
  for k in 1 3 5; do for ov in 0.25 1.0; do job dense-qt $k $ov $w $s & done; done
  job lev 3 1.0 $w $s & job dense-q 3 1.0 $w $s &
done; done
for j in $(jobs -p); do [ $j != $VLLM ] && wait $j; done
kill $VLLM; sleep 5
for pid in $(ps -eo pid,args | grep "[v]llm serve" | awk '{print $1}'); do kill $pid; done
echo "[$(date +%T)] DISP_STRESS_DONE (vLLM stopped)" >> $LOG
