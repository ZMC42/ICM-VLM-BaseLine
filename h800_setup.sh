#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-ICM-VLM}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VLLM_DIR="${VLLM_DIR:-${ROOT_DIR}/dependencies/vllm}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.10.0}"
MAX_JOBS="${MAX_JOBS:-8}"
NVCC_THREADS="${NVCC_THREADS:-1}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
WHEEL_DIR="${WHEEL_DIR:-${ROOT_DIR}/torch_cu126_wheels}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found" >&2
  exit 1
fi

if [ ! -d "${VLLM_DIR}" ]; then
  echo "vLLM source directory not found: ${VLLM_DIR}" >&2
  exit 1
fi

if [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
  echo "nvcc not found under CUDA_HOME=${CUDA_HOME}" >&2
  echo "Set CUDA_HOME to the local CUDA 12.6 toolkit path before running this script." >&2
  exit 1
fi

if ! command -v gcc >/dev/null 2>&1; then
  echo "gcc not found" >&2
  exit 1
fi

if ! command -v g++ >/dev/null 2>&1; then
  echo "g++ not found" >&2
  exit 1
fi

if ! command -v ninja >/dev/null 2>&1; then
  echo "ninja not found in PATH; it will be installed into the conda env, but a system ninja is preferred for large source builds." >&2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${ENV_NAME}"

python -m pip install -U pip

if [ -d "${WHEEL_DIR}" ] && find "${WHEEL_DIR}" -maxdepth 1 -type f -name 'torch-*.whl' | grep -q .; then
  echo "Using offline wheels from ${WHEEL_DIR}"
  python -m pip install --no-index --find-links "${WHEEL_DIR}" \
    numpy packaging setuptools wheel cmake ninja openai httpx
  python -m pip install --no-index --find-links "${WHEEL_DIR}" \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}"
else
  echo "Offline wheel dir not found or incomplete, falling back to online install"
  python -m pip install -U setuptools wheel packaging numpy cmake ninja
  python -m pip install -U "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" --index-url https://download.pytorch.org/whl/cu126
  python -m pip install -U openai httpx
fi

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS
export NVCC_THREADS
export VLLM_USE_PRECOMPILED=0

python - <<'PY'
import os
import shutil
import subprocess
import sys

checks = [
    ("python", sys.executable),
    ("nvcc", shutil.which("nvcc") or "missing"),
    ("gcc", shutil.which("gcc") or "missing"),
    ("g++", shutil.which("g++") or "missing"),
    ("ninja", shutil.which("ninja") or "missing"),
]
for key, value in checks:
    print(f"{key} = {value}")

print("cuda_home =", os.environ.get("CUDA_HOME"))
print("max_jobs =", os.environ.get("MAX_JOBS"))
print("nvcc_threads =", os.environ.get("NVCC_THREADS"))

for cmd in ([os.path.join(os.environ["CUDA_HOME"], "bin", "nvcc"), "--version"], ["nvidia-smi"]):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)
PY

python -m pip install -e "${VLLM_DIR}" --no-build-isolation

python - <<'PY'
import os
import torch
import vllm
print('cuda_home =', os.environ.get('CUDA_HOME'))
print('torch =', torch.__version__)
print('cuda(runtime) =', torch.version.cuda)
print('gpu =', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
print('vllm =', vllm.__version__)
print('vllm_file =', vllm.__file__)
if '+cu126' not in torch.__version__:
    raise SystemExit(f'Expected torch cu126 build, got {torch.__version__}')
if torch.version.cuda != '12.6':
    raise SystemExit(f'Expected torch CUDA runtime 12.6, got {torch.version.cuda}')
PY

echo
echo "H800 source-build environment is ready."
echo "Run: bash ${ROOT_DIR}/h800_server.sh"
