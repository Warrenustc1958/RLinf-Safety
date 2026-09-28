#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "$SCRIPT_DIR")")"
WORKSPACE_ROOT="$(dirname "$REPO_PATH")"
VENV_PATH="${VENV_PATH:-${WORKSPACE_ROOT}/.venv-openvlaoft-libero-safety}"

if [ ! -f "${VENV_PATH}/bin/activate" ]; then
  echo "Environment not found: ${VENV_PATH}" >&2
  exit 1
fi
source "${VENV_PATH}/bin/activate"
cd "$REPO_PATH"

export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATASET="${DATASET:-${WORKSPACE_ROOT}/datasets/libero_safety}"
MODEL="${MODEL:-${WORKSPACE_ROOT}/checkpoints/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora}"
LORA_PATH="${LORA_PATH:-${MODEL}/lora_adapter}"

exec python examples/sft/train_vla_sft.py \
  --config-name libero_safety_sft_openvlaoft \
  "data.train_data_paths=${DATASET}" \
  "actor.model.model_path=${MODEL}" \
  "actor.model.lora_path=${LORA_PATH}" \
  "$@"
