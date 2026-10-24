#!/bin/bash
P=~/Memloop; cd $P
LOG=logs/supp9_round2.log
note() { echo "[$(date +%T)] $*" >> $LOG; }
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=EMPTY HF_HUB_OFFLINE=1
export RUN_TAG=qwen3-14b-main
ARGS="--propagate --inject-num 5 --num-benign 20 --test-num 15 --eval-num 20"
nohup serve/vllm_qwen3_14b.sh > logs/vllm_supp9.log 2>&1 < /dev/null &
VLLM=$!
until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done
for p in 3005 3006; do ss -ltn | grep -q ":$p " || bash scripts/webshop_instance.sh $p >> $LOG; done
note "vLLM and WebShop 3005/3006 up"
job() {
  T=$1; PORT=$2; export WEBSHOP_URL=http://localhost:$PORT/
  NEW=recovered_contain_sim_${T}_r2
  note "$T round 2 (:$PORT) start"
  WORK_SUFFIX=_sim_${T}_r2 scripts/run_minja_recovery.sh 0 2 recover $ARGS --variant contain \
    --seeding origin --from-state recovered_contain_sim_$T --flags detector_flags_sim_${T}_r2.json \
    --out-tag $NEW &&
  WORK_SUFFIX=_sim_${T}_r2 scripts/run_minja_recovery.sh 0 2 eval $ARGS --eval-on $NEW
  note "$T round 2 done (exit $?)"
}
job r075_f05_d0 3005 & A=$!
job r050_f05_d0 3006 & B=$!
wait $A $B
kill $VLLM; sleep 20
for pid in $(ps -u $USER -o pid,args | grep "[v]llm serve" | awk "{print \$1}"); do kill $pid; done
for pid in $(ps -u $USER -o pid,args | grep "[w]eb_agent_site.app" | awk "{print \$1}"); do kill $pid; done
note "SUPP9_DONE (servers stopped)"
