#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$SCRIPT_DIR")"
WORKSPACE_ROOT="$(dirname "$REPO_PATH")"
VENV_PATH="${VENV_PATH:-${WORKSPACE_ROOT}/.venv-openvlaoft-libero-safety}"
LIBERO_SAFETY_PATH="${LIBERO_SAFETY_PATH:-${WORKSPACE_ROOT}/LIBERO-Safety}"

if [ ! -f "${VENV_PATH}/bin/activate" ]; then
  echo "Environment not found: ${VENV_PATH}" >&2
  echo "Run the installation commands in SAFELIBERO_INTEGRATION.md first." >&2
  exit 1
fi
source "${VENV_PATH}/bin/activate"
cd "$REPO_PATH"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# The container's /tmp is a small (30 GB) filesystem and Ray refuses object
# spilling once it is 95% full. Keep Ray sessions and other temporary files on
# the large root filesystem by default. Override RLINF_RUNTIME_DIR when needed.
RLINF_RUNTIME_DIR="${RLINF_RUNTIME_DIR:-/root/.cache/rlinf/libero-safety}"
export RAY_TMPDIR="${RAY_TMPDIR:-${RLINF_RUNTIME_DIR}/ray}"
export TMPDIR="${TMPDIR:-${RLINF_RUNTIME_DIR}/tmp}"
mkdir -p "$RAY_TMPDIR" "$TMPDIR"
export LIBERO_TYPE=safety
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${VENV_PATH}/share/libero-safety}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export PYTHONPATH="${LIBERO_SAFETY_PATH}:${REPO_PATH}:${PYTHONPATH:-}"

if [ ! -d "${LIBERO_SAFETY_PATH}/libero/libero/assets" ]; then
  echo "LIBERO-Safety assets are missing under ${LIBERO_SAFETY_PATH}/libero/libero/assets" >&2
  echo "Download assets.zip as described in SAFELIBERO_INTEGRATION.md." >&2
  exit 1
fi

MODE="${MODE:-eval}"
SUITE="${SUITE:-obstacle_avoidance}"
LEVEL="${LEVEL:-L0}"
GPU_RANKS="${GPU_RANKS:-0-3}"
TOTAL_ENVS="${TOTAL_ENVS:-250}"
MODEL="${MODEL:-${WORKSPACE_ROOT}/checkpoints/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora}"
UNNORM_KEY="${UNNORM_KEY:-libero_130_no_noops_trajall}"
IS_LORA="${IS_LORA:-True}"
LORA_PATH="${LORA_PATH:-${MODEL}/lora_adapter}"
EXTRA_OVERRIDES=("$@")

case "$SUITE" in
  affordance|human_safety|obstacle_avoidance|obstacle_avoidance_human) ;;
  *)
    echo "Unknown SUITE=${SUITE}" >&2
    echo "Choose affordance, human_safety, obstacle_avoidance, or obstacle_avoidance_human." >&2
    exit 2
    ;;
esac
case "${LEVEL^^}" in L0|L1|L2) ;; *) echo "LEVEL must be L0, L1, or L2" >&2; exit 2 ;; esac

common_model_args=(
  "rollout.model.model_path=${MODEL}"
  "rollout.model.unnorm_key=${UNNORM_KEY}"
  "rollout.model.is_lora=${IS_LORA}"
)
if [[ "${IS_LORA,,}" == "true" ]]; then
  common_model_args+=("rollout.model.lora_path=${LORA_PATH}")
fi

if [ "$MODE" = "eval" ]; then
  exec bash evaluations/run_eval.sh libero libero_safety_openvlaoft_eval \
    '~cluster.component_placement' \
    "+cluster.component_placement.env=${GPU_RANKS}" \
    "+cluster.component_placement.rollout=${GPU_RANKS}" \
    "env.eval.task_suite_name=${SUITE}" \
    "env.eval.safety_level=${LEVEL^^}" \
    "env.eval.total_num_envs=${TOTAL_ENVS}" \
    "${common_model_args[@]}" \
    "${EXTRA_OVERRIDES[@]}"
elif [ "$MODE" = "rollout" ]; then
  actor_model_args=(
    "actor.model.model_path=${MODEL}"
    "actor.model.unnorm_key=${UNNORM_KEY}"
    "actor.model.is_lora=${IS_LORA}"
  )
  if [[ "${IS_LORA,,}" == "true" ]]; then
    actor_model_args+=("actor.model.lora_path=${LORA_PATH}")
  fi
  exec python examples/embodiment/train_embodied_agent.py \
    --config-path "${EMBODIED_PATH}/config/" \
    --config-name libero_safety_grpo_openvlaoft \
    '~cluster.component_placement' \
    "+cluster.component_placement.actor=${GPU_RANKS}" \
    "+cluster.component_placement.env=${GPU_RANKS}" \
    "+cluster.component_placement.rollout=${GPU_RANKS}" \
    "env.train.task_suite_name=${SUITE}" \
    "env.eval.task_suite_name=${SUITE}" \
    "env.train.safety_level=${LEVEL^^}" \
    "env.eval.safety_level=${LEVEL^^}" \
    "${common_model_args[@]}" \
    "${actor_model_args[@]}" \
    "${EXTRA_OVERRIDES[@]}"
else
  echo "MODE must be eval or rollout" >&2
  exit 2
fi
