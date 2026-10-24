#!/bin/bash
P=~/Memloop; cd $P
LOG=logs/supp6_pilot.log
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=EMPTY OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=~/miniconda3/envs/memsec/bin/python
note() { echo "[$(date +%T)] $*" >> $LOG; }
for pid in $(ps -u $USER -o pid,args | grep "[v]llm serve" | awk '{print $1}'); do kill $pid; done; sleep 10
nohup serve/vllm_qwen3_14b_bi.sh > logs/vllm_supp6_bi.log 2>&1 < /dev/null &
until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done
note "batch-invariant vLLM up"
OUT=$P/results/amem_evo_pilot/k3/ov1.0/seed0; mkdir -p $OUT
A="--subject nutrition --seed 0 --model qwen3-14b --propagate --num-templates 5 --num-benign 60
   --max-benign-retries 3 --num-test 10 --retriever amem --evolve --n-shots 3 --overlap 1.0
   --attack-window 0.55 0.95 --out $OUT"
r() { $PY memloop/experiments/qa_recovery.py "$@" $A >> $OUT/${1}${V:-}.log 2>&1; }
r attack || note "attack FAILED"; note "attack done"
r recover & a=$!; V=_b3 r recover --variant b3 & b=$!; r counterfactual & c=$!
wait $a; note "recover exit $?"; wait $b; note "b3 exit $?"; wait $c; note "cf exit $?"
r compare || note "compare FAILED"
r baselines; r eval --eval-on attack B1 B2 recovered_contain recovered_contain_b3 counterfactual
note "eval done"
for pid in $(ps -u $USER -o pid,args | grep "[v]llm serve" | awk '{print $1}'); do kill $pid; done
note "SUPP6_PILOT_DONE (vLLM stopped)"
