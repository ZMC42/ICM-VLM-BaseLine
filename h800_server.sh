#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-ICM-VLM}"
MODEL="${MODEL:-ModelHub/Qwen/Qwen3.5-27B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
fi

vllm serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --dtype "${DTYPE}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --attention-config '{"backend":"FLASH_ATTN","flash_attn_version":2}' \
  --enforce-eager \
  --reasoning-parser qwen3 \
  --language-model-only
