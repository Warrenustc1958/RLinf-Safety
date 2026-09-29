#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$SCRIPT_DIR")"
WORKSPACE_ROOT="$(dirname "$REPO_PATH")"
VENV_PATH="${VENV_PATH:-${WORKSPACE_ROOT}/.venv-openpi-libero-safety}"
LIBERO_SAFETY_PATH="${LIBERO_SAFETY_PATH:-${WORKSPACE_ROOT}/LIBERO-Safety}"
MODEL="${MODEL:-${WORKSPACE_ROOT}/checkpoints/pi05_libero_safety_rlinf}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/oss/xujingbo/evaluations/libero_safety/pi05_rlinf}"

if [ ! -f "${VENV_PATH}/bin/activate" ]; then
  echo "OpenPI environment not found: ${VENV_PATH}" >&2
  exit 1
fi
if [ ! -f "${MODEL}/model.safetensors" ]; then
  echo "Converted RLinf checkpoint not found: ${MODEL}/model.safetensors" >&2
  echo "Convert the Orbax checkpoint with jax_to_openpi_rlinf first." >&2
  exit 1
fi
if [ ! -f "${MODEL}/assets/lerobot/norm_stats.json" ]; then
  echo "Normalization stats not found under ${MODEL}/assets/lerobot" >&2
  exit 1
fi

source "${VENV_PATH}/bin/activate"
cd "$REPO_PATH"

export LIBERO_TYPE=safety
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${WORKSPACE_ROOT}/.venv-openvlaoft-libero-safety/share/libero-safety}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export PYTHONPATH="${LIBERO_SAFETY_PATH}:${REPO_PATH}:${PYTHONPATH:-}"

# Ray appends a long session/socket suffix. Keep this prefix short enough for
# Linux's 107-byte AF_UNIX path limit.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/rp}"
export TMPDIR="${TMPDIR:-/tmp/p5}"
mkdir -p "$RAY_TMPDIR" "$TMPDIR" "$OUTPUT_ROOT"

MODE="${MODE:-smoke}"
SUITE="${SUITE:-obstacle_avoidance}"
LEVEL="${LEVEL:-L0}"
GPU_RANKS="${GPU_RANKS:-0}"
TOTAL_ENVS="${TOTAL_ENVS:-50}"
EVAL_SEED="${EVAL_SEED:-0}"
EXTRA_OVERRIDES=("$@")

case "$SUITE" in
  affordance|human_safety|obstacle_avoidance|obstacle_avoidance_human) ;;
  *) echo "Unknown SUITE=${SUITE}" >&2; exit 2 ;;
esac
case "${LEVEL^^}" in L0|L1|L2) ;; *) echo "LEVEL must be L0, L1, or L2" >&2; exit 2 ;; esac
case "$MODE" in smoke|cell|all) ;; *) echo "MODE must be smoke, cell, or all" >&2; exit 2 ;; esac

run_cell() {
  local suite="$1"
  local level="$2"
  local total_envs="$3"
  local max_episode_steps="$4"
  local max_rollout_steps="$5"
  local task_filter="$6"
  local run_name="${suite}_${level}"
  local run_dir="${OUTPUT_ROOT}/${run_name}"
  local task_filter_arg=()

  if [ -n "$task_filter" ]; then
    task_filter_arg=("env.eval.task_id_filter=${task_filter}")
  fi
  mkdir -p "$run_dir"
  echo "Evaluating Pi0.5: suite=${suite}, level=${level}, envs=${total_envs}"

  bash evaluations/run_eval.sh libero libero_safety_openpi_pi05_rlinf_eval \
    '~cluster.component_placement' \
    "+cluster.component_placement.env=${GPU_RANKS}" \
    "+cluster.component_placement.rollout=${GPU_RANKS}" \
    "env.eval.task_suite_name=${suite}" \
    "env.eval.safety_level=${level}" \
    "env.eval.total_num_envs=${total_envs}" \
    "env.eval.max_episode_steps=${max_episode_steps}" \
    "env.eval.max_steps_per_rollout_epoch=${max_rollout_steps}" \
    "env.eval.seed=${EVAL_SEED}" \
    "rollout.model.model_path=${MODEL}" \
    "runner.logger.log_path=${run_dir}" \
    "runner.logger.experiment_name=pi05_libero_safety_${run_name}" \
    "${task_filter_arg[@]}" \
    "${EXTRA_OVERRIDES[@]}"
}

if [ "$MODE" = smoke ]; then
  run_cell "$SUITE" "${LEVEL^^}" 1 10 10 '[0]'
elif [ "$MODE" = cell ]; then
  run_cell "$SUITE" "${LEVEL^^}" "$TOTAL_ENVS" 512 515 ''
else
  for suite in affordance human_safety obstacle_avoidance obstacle_avoidance_human; do
    for level in L0 L1 L2; do
      run_cell "$suite" "$level" "$TOTAL_ENVS" 512 515 ''
    done
  done
fi
