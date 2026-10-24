#!/bin/bash
P=~/Memloop; cd $P
LOG=logs/supp6_evo.log
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=EMPTY OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=~/miniconda3/envs/memsec/bin/python
note() { echo "[$(date +%T)] $*" >> $LOG; }
for pid in $(ps -u $USER -o pid,args | grep "[v]llm serve" | awk '{print $1}'); do kill $pid; done; sleep 10
nohup serve/vllm_qwen3_14b_bi.sh > logs/vllm_supp6_bi.log 2>&1 < /dev/null &
VLLM=$!
until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done
note "batch-invariant vLLM up"
job() {
  local OUT=$P/results/amem_evo/w$4/k$1/ov$2/seed$3
  mkdir -p $OUT
  local PL="--num-templates 5 --num-benign 60 --attack-window 0.55 0.95"
  [ $4 = 80 ] && PL="--num-templates 2 --num-benign 120 --attack-window 0.80 0.95"
  local A="--subject nutrition --seed $3 --model qwen3-14b --propagate $PL
           --max-benign-retries 3 --num-test 10 --retriever amem --evolve --n-shots $1 --overlap $2
           --out $OUT"
  r() { $PY memloop/experiments/qa_recovery.py "$@" $A >> $OUT/${1}${V:-}.log 2>&1; }
  [ -f $OUT/attack.pkl ] || r attack || { note "w$4/k$1/ov$2/seed$3 attack FAILED"; return; }
  [ -f $OUT/recovered_contain.pkl ] || r recover & a=$!
  [ -f $OUT/recovered_contain_b3.pkl ] || V=_b3 r recover --variant b3 & b=$!
  [ -f $OUT/counterfactual.pkl ] || r counterfactual & c=$!
  wait $a $b $c
  r compare || note "w$4/k$1/ov$2/seed$3 compare FAILED"
  [ -f $OUT/eval_attack_B1_B2_recovered_contain_recovered_contain_b3_counterfactual.json ] || { r baselines; r eval --eval-on attack B1 B2 recovered_contain recovered_contain_b3 counterfactual; }
  note "w$4/k$1/ov$2/seed$3 done"
}
for w in 55 80; do for s in 0 1 2; do for k in 1 3 5; do for ov in 0.25 1.0; do
  job $k $ov $s $w & J="$J $!"; done; done; done; done
wait $J
kill $VLLM; sleep 5
for pid in $(ps -u $USER -o pid,args | grep "[v]llm serve" | awk '{print $1}'); do kill $pid; done
note "SUPP6_EVO_DONE (vLLM stopped)"
