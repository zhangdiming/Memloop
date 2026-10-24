#!/bin/bash
P=~/Memloop; cd $P
LOG=logs/supp2.log
note() { echo "[$(date +%T)] $*" >> $LOG; }
stop_vllm() { for pid in $(ps -u $USER -o pid,args | grep "[v]llm serve" | awk '{print $1}'); do kill $pid; done; sleep 30; }
webshop() { for p in "$@"; do ss -ltn | grep -q ":$p " || bash scripts/webshop_instance.sh $p >> $LOG; done; }

stop_vllm
nohup serve/vllm_qwen3_14b_bi.sh > logs/vllm_supp2_bi.log 2>&1 < /dev/null &
until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done
webshop 3000 3001 3002
note "phase 1: batch-invariant vLLM up, WebShop 3000-3002 ready"

det2() {
  local IDX=$1; export WEBSHOP_URL=http://localhost:$2/ RUN_TAG=qwen3-14b-det2
  local ARGS="--propagate --inject-num 5 --num-benign 20 --test-num 15 --eval-num 10"
  run() { scripts/run_minja_recovery.sh $IDX 0 "$@" $ARGS; }
  note "det2 $IDX start"
  run attack || { note "det2 $IDX attack FAILED"; return; }
  WORK_SUFFIX=_cfd run counterfactual --cf-tag counterfactual_during6 --during 6 & a=$!
  WORK_SUFFIX=_rd run recover --variant contain --during 6 & b=$!
  wait $a $b
  run compare --base counterfactual_during6
  note "det2 $IDX done"
}
qa() {
  export OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=EMPTY OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  local PY=~/miniconda3/envs/memsec/bin/python OUT=$P/results/amem_qa/k$1/seed$2
  mkdir -p $OUT
  local A="--subject nutrition --seed $2 --model qwen3-14b --propagate --num-templates 5 --num-benign 60
           --max-benign-retries 3 --num-test 10 --retriever amem --n-shots $1 --out $OUT"
  r() { $PY memloop/experiments/qa_recovery.py "$@" $A >> $OUT/${1}${V:-}.log 2>&1; }
  r attack || { note "amem k$1 seed$2 attack FAILED"; return; }
  r recover & a=$!
  V=_b3 r recover --variant b3 & b=$!
  r counterfactual & c=$!
  wait $a $b $c
  r compare
  note "amem k$1 seed$2 done"
}
( IDX=8; det2 8 3000 ) & j1=$!
( IDX=5; det2 5 3001 ) & j2=$!
( IDX=2; det2 2 3002 ) & j3=$!
( for s in 0 1 2 3 4; do qa 3 $s & qa 5 $s & wait; done ) & j4=$!
wait $j1 $j2 $j3 $j4
note "PHASE1_DONE"

stop_vllm
nohup serve/vllm_qwen3_14b.sh > logs/vllm_supp2_default.log 2>&1 < /dev/null &
until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done
webshop 3000 3001 3002 3003 3004
note "phase 2: default vLLM up, WebShop 3000-3004 ready"
M=results/minja_recovery/qwen3-14b-main
for f in toothbrush/seed0/recovered_contain_entry camera/seed2/recovered_contain_union_k2; do
  for x in $M/$f.pkl $M/${f}_stats.json; do [ -f $x ] && mv $x ${x%.*}_oldadapter.${x##*.}; done
done
{
  for s in 0 1 2; do for i in 0 1 2 3 4 5 6 7 8; do echo "ent $i $s"; done; done
  echo "k2u 0 2"
} | xargs -P 5 -L 1 bash scripts/supp2_job.sh
stop_vllm
for pid in $(ps -u $USER -o pid,args | grep "[w]eb_agent_site.app" | awk '{print $1}'); do kill $pid; done
note "SUPP2_DONE (servers stopped)"
