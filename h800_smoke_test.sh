#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-ICM-VLM}"
MODEL="${MODEL:-ModelHub/Qwen/Qwen3.5-9B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18000}"
TP_SIZE="${TP_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
LOG_FILE="${LOG_FILE:-${ROOT_DIR}/h800_smoke_test.log}"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

python - <<'PY'
import torch
import vllm
print('torch =', torch.__version__)
print('cuda(runtime) =', torch.version.cuda)
print('gpu =', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
print('vllm =', vllm.__version__)
print('vllm_file =', vllm.__file__)
PY

: > "${LOG_FILE}"
nohup vllm serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --dtype "${DTYPE}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --attention-config '{"backend":"FLASH_ATTN","flash_attn_version":2}' \
  --enforce-eager \
  --reasoning-parser qwen3 \
  --language-model-only \
  > "${LOG_FILE}" 2>&1 &
SERVER_PID=$!

echo "Started vLLM smoke test server pid=${SERVER_PID}, log=${LOG_FILE}"

python - <<'PY' "${HOST}" "${PORT}" "${STARTUP_TIMEOUT}"
import json
import sys
import time
from urllib import error, request

host = sys.argv[1]
port = sys.argv[2]
timeout = int(sys.argv[3])
url = f'http://{host}:{port}/v1/models'
last_error = None
for _ in range(timeout):
    try:
        with request.urlopen(url, timeout=5) as resp:
            if resp.status == 200:
                print('Server is ready')
                raise SystemExit(0)
    except Exception as exc:
        last_error = exc
        time.sleep(1)
print(f'Server did not become ready within {timeout}s: {last_error}', file=sys.stderr)
raise SystemExit(1)
PY

python - <<'PY' "${HOST}" "${PORT}" "${MODEL}"
import json
import sys
from urllib import request

host = sys.argv[1]
port = sys.argv[2]
model = sys.argv[3]
url = f'http://{host}:{port}/v1/chat/completions'
payload = {
    'model': model,
    'messages': [{'role': 'user', 'content': '请用一句话回答：你已成功启动。'}],
    'temperature': 0.0,
    'max_tokens': 64,
    'chat_template_kwargs': {'enable_thinking': False},
}
req = request.Request(
    url,
    data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
    headers={'Content-Type': 'application/json'},
    method='POST',
)
with request.urlopen(req, timeout=300) as resp:
    body = json.loads(resp.read().decode('utf-8'))
msg = body['choices'][0]['message']
content = msg.get('content')
reasoning = msg.get('reasoning')
print('response_content =', content)
if reasoning is not None:
    print('response_reasoning =', reasoning)
if not content and not reasoning:
    raise SystemExit('Smoke test request returned empty content and reasoning')
PY

echo "Smoke test passed."
echo "Recent server log:"
tail -n 40 "${LOG_FILE}" || true
