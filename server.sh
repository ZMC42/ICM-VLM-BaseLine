#!/bin/bash

vllm serve ModelHub/Qwen/Qwen3.5-9B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-cudagraph-capture-size 8 \
  --reasoning-parser qwen3 \
  --language-model-only
