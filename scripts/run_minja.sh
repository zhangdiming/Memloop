#!/bin/bash
set -eo pipefail
IDX=${1:?victim index}; SEED=${2:-0}; MODEL=${3:-qwen3-14b}
P=~/Memloop; RAP=$P/baselines/MINJA/rap
source ~/miniconda3/bin/activate minja-rap
set -u
read -r VICTIM TARGET PRICE < <(python -c "
import json; x = json.load(open('$RAP/victim_target_pair/victim_target.json'))[$IDX]
print(x['victim'].replace(' ', '_'), x['target'].replace(' ', '\x1f'), x['target_price'])")
VICTIM=${VICTIM//_/ }; TARGET=${TARGET//$'\x1f'/ }
TAG=${VICTIM// /_}
OUT=$P/results/minja_rap/$MODEL/$TAG/seed$SEED
WORK=$P/workdirs/minja_rap/$MODEL/$TAG/seed$SEED
mkdir -p $OUT $WORK/log
for f in minja.py configs prompts webshop_instructions.json indication_prompt_template.json victim_target_pair; do
  ln -sfn $RAP/$f $WORK/$f
done
cd $WORK
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=EMPTY MINJA_SEED=$SEED
if [ "${MINJA_EMB_DEVICE:-}" = cpu ]; then export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}; fi
python minja.py --model_name $MODEL --seed $SEED --victim "$VICTIM" --target "$TARGET" \
  --target_price $PRICE --memory_output $OUT/memory.json --test_memory_output $OUT/test_memory.json
cp "$(ls -t log/program_output_*.log | head -1)" $OUT/run.log
cp merged_instructions.json $OUT/
echo "emb_device=${MINJA_EMB_DEVICE:-default}" > $OUT/run_meta.txt
grep -E "inject success rate|attack success rate" $OUT/run.log
