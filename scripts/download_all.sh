#!/bin/bash
set -u
P=~/Memloop; E=~/miniconda3/envs; HF=$E/memsec/bin/hf
cd $P
echo "[$(date)] start"
for m in facebook/contriever facebook/contriever-msmarco facebook/dpr-ctx_encoder-single-nq-base \
         facebook/dpr-question_encoder-single-nq-base castorini/ance-dpr-question-multi BAAI/bge-large-en \
         BAAI/bge-base-en-v1.5 google-bert/bert-base-uncased sentence-transformers/all-MiniLM-L6-v2 \
         sentence-transformers/all-MiniLM-L12-v2 sentence-transformers/msmarco-roberta-base-ance-firstp; do
  $HF download $m --exclude "*.h5" "*.msgpack" "*.ot" "*onnx*" "*openvino*" > /dev/null 2>&1 && echo "OK model $m" || echo "FAIL model $m"
done
for m in Qwen/Qwen3-8B Qwen/Qwen3-14B; do
  $HF download $m > /dev/null 2>&1 && echo "OK model $m" || echo "FAIL model $m"
done
mkdir -p data/longmemeval
for f in longmemeval_oracle.json longmemeval_s_cleaned.json longmemeval_m_cleaned.json; do
  $HF download xiaowu0162/longmemeval-cleaned $f --repo-type dataset --local-dir data/longmemeval > /dev/null 2>&1 && echo "OK data $f" || echo "FAIL data $f"
done
ln -sfn $P/data/longmemeval baselines/LongMemEval/data
cd baselines/PoisonedRAG && $E/poisonedrag/bin/python prepare_dataset.py > /dev/null 2>&1 && echo "OK data beir $(ls datasets)" || echo "FAIL data beir"
cd $P
mkdir -p data/strategyqa && cd data/strategyqa && wget -q https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip && unzip -oq strategyqa_dataset.zip && echo "OK data strategyqa" || echo "FAIL data strategyqa"
echo "[$(date)] DOWNLOAD_DONE"
