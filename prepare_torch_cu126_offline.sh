#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEEL_DIR="${WHEEL_DIR:-${ROOT_DIR}/torch_cu126_wheels}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.10.0}"

mkdir -p "${WHEEL_DIR}"

"${PYTHON_BIN}" -m pip download -d "${WHEEL_DIR}" \
  --index-url https://download.pytorch.org/whl/cu126 \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}"

"${PYTHON_BIN}" -m pip download -d "${WHEEL_DIR}" \
  numpy packaging setuptools wheel cmake ninja openai httpx

cat > "${WHEEL_DIR}/README.offline.txt" <<TXT
Offline wheels prepared for H800 deployment.

Install on H800 with:

pip install --no-index --find-links ${WHEEL_DIR} \
  torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION}

pip install --no-index --find-links ${WHEEL_DIR} \
  numpy packaging setuptools wheel cmake ninja openai httpx
TXT

echo "Prepared offline wheels under: ${WHEEL_DIR}"
ls -1 "${WHEEL_DIR}"
