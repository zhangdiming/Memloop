#!/bin/bash
set -eo pipefail
IDX=${1:?victim index}; SEED=${2:?seed}; PHASE=${3:?phase}; shift 3
MODEL=${MODEL:-qwen3-14b}
P=~/Memloop; RAP=$P/baselines/MINJA/rap
source ~/miniconda3/bin/activate minja-rap
set -u
TAG=$(python -c "import json; print(json.load(open('$RAP/victim_target_pair/victim_target.json'))[$IDX]['victim'].replace(' ', '_'))")
OUT=$P/results/minja_recovery/${RUN_TAG:-$MODEL}/$TAG/seed$SEED
WORK=$P/workdirs/minja_recovery/${RUN_TAG:-$MODEL}/$TAG/seed$SEED/$PHASE${WORK_SUFFIX:-}
mkdir -p $OUT $WORK
for f in minja.py configs prompts webshop_instructions.json indication_prompt_template.json victim_target_pair; do
  ln -sfn $RAP/$f $WORK/$f
done
cd $WORK
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=EMPTY
export MINJA_TEMPERATURE=0 MINJA_SEED=$SEED
python $P/memloop/experiments/minja_recovery.py $PHASE --victim-idx $IDX --seed $SEED \
  --model $MODEL --out $OUT "$@" > $OUT/${PHASE}${WORK_SUFFIX:-}_agent.log 2> >(tee -a $OUT/${PHASE}${WORK_SUFFIX:-}.log >&2)
