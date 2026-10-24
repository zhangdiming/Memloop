#!/bin/bash
source ~/miniconda3/bin/activate memsec
export VLLM_USE_FLASHINFER_SAMPLER=0
exec vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization ${GPU_UTIL:-0.85} \
  --enable-prefix-caching --seed 0 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
