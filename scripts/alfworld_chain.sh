#!/bin/bash
P=~/Memloop; cd $P
M=results/alfworld/qwen3-14b-main
until [ "$(ls $M/*/seed*/eval_attack_B1_B2_DP_recovered_contain_recovered_b3_counterfactual.json 2>/dev/null | wc -l)" -ge 27 ] \
      && ! ps -u $USER -o args | grep -q "[a]lfworld_job.sh"; do sleep 120; done
echo "[$(date +%T)] sweep complete; detector" >> logs/alfworld_sweep.log
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=EMPTY \
  ~/miniconda3/envs/alfworld/bin/python memloop/experiments/detect_llm_alfworld.py > logs/alfworld_detect.log 2>&1
python3 scripts/alfworld_summary.py > logs/alfworld_summary.txt 2>&1
echo "[$(date +%T)] detector done; deterministic subset" >> logs/alfworld_sweep.log
bash scripts/alfworld_det.sh
