#!/bin/bash
IDX=$1; SEED=$2; PHASE=$3; shift 3
P=~/Memloop; cd $P
export ALFWORLD_DATA=$P/data/alfworld HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1} OPENAI_API_KEY=EMPTY
export RAP_TEMPERATURE=${RAP_TEMPERATURE:-0} RAP_SEED=$SEED
RUN_TAG=${RUN_TAG:-qwen3-14b-main}
V=$(python3 -c "import json;print(json.load(open('memloop/experiments/alfworld_pairs.json'))[$IDX]['victim'])")
OUT=results/alfworld/$RUN_TAG/$V/seed$SEED; mkdir -p $OUT
exec ~/miniconda3/envs/alfworld/bin/python memloop/experiments/alfworld_recovery.py $PHASE \
  --pair-idx $IDX --seed $SEED --out $OUT "$@" >> $OUT/${PHASE}${WORK_SUFFIX}.log 2>&1
