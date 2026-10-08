#!/usr/bin/env bash
set -Eeuo pipefail

# Run this file inside a fresh tmux window on the OpenBayes host:
#   bash evaluations/run_pi05_libero_safety_2xh200_collect.sh
#
# --no-flash-attn is an installation-time option. The Pi0.5 evaluation path
# uses PyTorch SDPA and needs no runtime flash-attn override.

# ---------------------------------------------------------------------------
# 1. Current OpenBayes paths
# ---------------------------------------------------------------------------
export RLINF_ROOT="${RLINF_ROOT:-/openbayes/input/input2/RLinf-Safety}"
export LIBERO_SAFETY_PATH="${LIBERO_SAFETY_PATH:-/openbayes/input/input2/LIBERO-Safety}"
export LIBERO_DATASET_ROOT="${LIBERO_DATASET_ROOT:-/openbayes/input/input1/LIBERO-Safety}"
export PI05_SOURCE_PATH="${PI05_SOURCE_PATH:-/openbayes/input/input0/pi05_libero_safety}"
export WAN22_MODEL_PATH="${WAN22_MODEL_PATH:-/openbayes/input/input0/Wan2.2-TI2V-5B}"

export RUNTIME_ROOT="${RUNTIME_ROOT:-/openbayes/home/rlinf-safety-runtime}"
export VENV_PATH="${VENV_PATH:-${RUNTIME_ROOT}/.venv-openpi-libero-safety}"
# This must be the converted RLinf/OpenPI checkpoint, not the source Orbax tree.
export MODEL="${MODEL:-${RUNTIME_ROOT}/checkpoints/pi05_libero_safety_rlinf}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${RUNTIME_ROOT}/rollouts/pi05_libero_safety_four_quadrant/${RUN_ID}}"
export DATASET_OUTPUT_ROOT="${DATASET_OUTPUT_ROOT:-${RUNTIME_ROOT}/datasets/pi05_libero_safety_four_quadrant/${RUN_ID}}"

# ---------------------------------------------------------------------------
# 2. Two-H200 runtime and download mirrors
# ---------------------------------------------------------------------------
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export GPU_RANKS="${GPU_RANKS:-0-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export LIBERO_TYPE=safety
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${VENV_PATH}/share/libero-safety}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export RLINF_LIBERO_DEBUG="${RLINF_LIBERO_DEBUG:-1}"
# Conservative workaround for NVIDIA/EGL drivers that abort in
# mjr_readPixels when multiple simulator subprocesses render concurrently.
# Set USE_GLOBAL_EGL_LOCK=0 after the run is stable to recover throughput.
export USE_GLOBAL_EGL_LOCK="${USE_GLOBAL_EGL_LOCK:-1}"
if [[ "${USE_GLOBAL_EGL_LOCK}" == "1" ]]; then
  export RLINF_LIBERO_EGL_LOCK_PATH="${RLINF_LIBERO_EGL_LOCK_PATH:-/tmp/rlinf-libero-egl.lock}"
else
  unset RLINF_LIBERO_EGL_LOCK_PATH
fi
export EMBODIED_PATH="${RLINF_ROOT}/examples/embodiment"
export PYTHONPATH="${LIBERO_SAFETY_PATH}:${RLINF_ROOT}:${PYTHONPATH:-}"

# Ray appends a long session suffix. A long /openbayes/... prefix exceeds the
# Linux AF_UNIX 107-byte limit, so this path is intentionally short.
unset RAY_ADDRESS
export RAY_TMPDIR="${SHORT_RAY_TMPDIR:-/tmp/rp-${RUN_ID}}"
export TMPDIR="${SHORT_TMPDIR:-/tmp/p5-${RUN_ID}}"

# Collection scale. MODE=all covers four physical suites x L0/L1/L2.
export TOTAL_ENVS="${TOTAL_ENVS:-50}"
export EVAL_SEED="${EVAL_SEED:-0}"
export RUN_SMOKE="${RUN_SMOKE:-1}"
export RUN_DUAL_GPU_PROBE="${RUN_DUAL_GPU_PROBE:-1}"
export PROBE_ONLY="${PROBE_ONLY:-0}"
export BUILD_DATASET_AFTER_ROLLOUT="${BUILD_DATASET_AFTER_ROLLOUT:-1}"
export RISK_WINDOW_STEPS="${RISK_WINDOW_STEPS:-32}"
export MAX_BRANCHES_PER_ROOT="${MAX_BRANCHES_PER_ROOT:-16}"
export ACTION_CHUNK="${ACTION_CHUNK:-1}"
if [[ "${ACTION_CHUNK}" != "1" ]]; then
  echo "Counterfactual collection requires ACTION_CHUNK=1, got ${ACTION_CHUNK}" >&2
  exit 2
fi
# One root plus all possible branches may each consume a 512-step horizon.
# Counterfactual collection deliberately executes one action per policy call
# so every restored state is observed before the branch action is predicted.
DEFAULT_MAX_ROLLOUT_STEPS=$(((MAX_BRANCHES_PER_ROOT + 1) * 512))
export MAX_ROLLOUT_STEPS="${MAX_ROLLOUT_STEPS:-${DEFAULT_MAX_ROLLOUT_STEPS}}"

# ---------------------------------------------------------------------------
# 3. Activate and validate the existing environment
# ---------------------------------------------------------------------------
test -f "${VENV_PATH}/bin/activate" || {
  echo "Missing venv: ${VENV_PATH}" >&2
  exit 1
}
source "${VENV_PATH}/bin/activate"

test -d "${LIBERO_SAFETY_PATH}/libero/libero/assets" || {
  echo "Missing LIBERO-Safety assets under ${LIBERO_SAFETY_PATH}/libero/libero/assets" >&2
  exit 1
}
test -f "${LIBERO_CONFIG_PATH}/config.yaml" || {
  echo "Missing LIBERO config: ${LIBERO_CONFIG_PATH}/config.yaml" >&2
  exit 1
}
test -f "${MODEL}/model.safetensors" || {
  echo "Missing converted Pi0.5 checkpoint: ${MODEL}/model.safetensors" >&2
  exit 1
}
test -f "${MODEL}/assets/lerobot/norm_stats.json" || {
  echo "Missing Pi0.5 normalization stats under ${MODEL}/assets/lerobot" >&2
  exit 1
}

for dataset_dir in meta video data; do
  if [[ ! -d "${LIBERO_DATASET_ROOT}/${dataset_dir}" ]]; then
    echo "Warning: demonstration directory is absent: ${LIBERO_DATASET_ROOT}/${dataset_dir}" >&2
  fi
done

mkdir -p "${RAY_TMPDIR}" "${TMPDIR}" "${OUTPUT_ROOT}"
exec > >(tee -a "${OUTPUT_ROOT}/launcher.log") 2>&1
cd "${RLINF_ROOT}"

echo "Run ID:               ${RUN_ID}"
echo "RLinf-Safety:         ${RLINF_ROOT}"
echo "LIBERO-Safety:        ${LIBERO_SAFETY_PATH}"
echo "LIBERO demonstrations:${LIBERO_DATASET_ROOT}"
echo "Pi0.5 source:         ${PI05_SOURCE_PATH}"
echo "Pi0.5 converted:      ${MODEL}"
echo "Wan2.2:               ${WAN22_MODEL_PATH}"
echo "Rollout output:       ${OUTPUT_ROOT}"
echo "Constructed dataset:  ${DATASET_OUTPUT_ROOT}"
echo "Ray temp:             ${RAY_TMPDIR}"
echo "Max rollout steps:    ${MAX_ROLLOUT_STEPS}"
echo "Action chunk:         ${ACTION_CHUNK}"
echo "Global EGL lock:      ${RLINF_LIBERO_EGL_LOCK_PATH:-disabled}"

nvidia-smi --query-gpu=index,name,memory.total --format=csv
python - <<'PY'
import ctypes
import torch

for library in ("libOpenGL.so.0", "libEGL.so.1", "libGLX.so.0"):
    ctypes.CDLL(library)
    print("loaded:", library)

count = torch.cuda.device_count()
print("visible CUDA devices:", count)
for index in range(count):
    print(index, torch.cuda.get_device_name(index))
if count != 2:
    raise SystemExit(f"Expected exactly two visible GPUs, got {count}")

import libero.libero as libero
from libero.libero.benchmark import get_benchmark

suite = get_benchmark("obstacle_avoidance")()
print("LIBERO module:", libero.__file__)
print("benchmark root:", libero.get_libero_path("benchmark_root"))
print("task distribution:", suite.get_task_distribution_by_level())
PY

COLLECT_OVERRIDES=(
  "rollout.model.num_action_chunks=${ACTION_CHUNK}"
  "env.eval.episode_auditor.enabled=true"
  "env.eval.episode_auditor.mode=shadow_collect"
  "env.eval.episode_auditor.include_observations=true"
  "env.eval.episode_auditor.save_visual_observations=true"
  "env.eval.episode_auditor.visual_camera_keys=[agentview_image]"
  "env.eval.episode_auditor.counterfactual_branch.enabled=true"
  "env.eval.episode_auditor.counterfactual_branch.risk_window_steps=${RISK_WINDOW_STEPS}"
  "env.eval.episode_auditor.counterfactual_branch.max_branches_per_root=${MAX_BRANCHES_PER_ROOT}"
  "env.eval.episode_auditor.counterfactual_branch.target_quadrants=[Q2,Q4]"
  "env.eval.episode_auditor.counterfactual_branch.perturbation_range=[0.15,1.0]"
  "env.eval.episode_auditor.counterfactual_branch.action_dimensions=[0,1,2,3,4,5]"
)

# ---------------------------------------------------------------------------
# 4. Fast end-to-end check, then the 12-cell collection
# ---------------------------------------------------------------------------
if [[ "${RUN_SMOKE}" == "1" ]]; then
  echo "Running a 10-step smoke test before the full collection..."
  ray stop --force || true
  MODE=smoke \
  SUITE=obstacle_avoidance \
  LEVEL=L0 \
  GPU_RANKS=0 \
  TOTAL_ENVS=1 \
  OUTPUT_ROOT="${OUTPUT_ROOT}/smoke" \
    bash evaluations/run_safelibero_pi05.sh "${COLLECT_OVERRIDES[@]}"
fi

if [[ "${RUN_DUAL_GPU_PROBE}" == "1" ]]; then
  echo "Running a 20-step two-GPU counterfactual/EGL probe..."
  ray stop --force || true
  MODE=cell \
  SUITE=affordance \
  LEVEL=L0 \
  GPU_RANKS="${GPU_RANKS}" \
  TOTAL_ENVS=2 \
  OUTPUT_ROOT="${OUTPUT_ROOT}/dual_gpu_probe" \
    bash evaluations/run_safelibero_pi05.sh \
      "${COLLECT_OVERRIDES[@]}" \
      "env.eval.max_episode_steps=20" \
      "env.eval.max_steps_per_rollout_epoch=20"
fi

if [[ "${PROBE_ONLY}" == "1" ]]; then
  echo "Single- and dual-GPU probes completed; PROBE_ONLY=1, stopping here."
  exit 0
fi

echo "Starting full four-quadrant collection: 4 suites x 3 levels..."
ray stop --force || true
MODE=all \
GPU_RANKS="${GPU_RANKS}" \
TOTAL_ENVS="${TOTAL_ENVS}" \
EVAL_SEED="${EVAL_SEED}" \
OUTPUT_ROOT="${OUTPUT_ROOT}/collection" \
  bash evaluations/run_safelibero_pi05.sh \
    "${COLLECT_OVERRIDES[@]}" \
    "env.eval.max_steps_per_rollout_epoch=${MAX_ROLLOUT_STEPS}"

# ---------------------------------------------------------------------------
# 5. Validate and construct the aligned Wan/SAM dataset
# ---------------------------------------------------------------------------
if [[ "${BUILD_DATASET_AFTER_ROLLOUT}" == "1" ]]; then
  mapfile -d '' AUDIT_DIRS < <(
    find "${OUTPUT_ROOT}/collection" -type d -name libero_safety_audits -print0
  )
  if (( ${#AUDIT_DIRS[@]} == 0 )); then
    echo "No libero_safety_audits directories were produced." >&2
    exit 1
  fi

  AUDIT_ARGS=()
  for audit_dir in "${AUDIT_DIRS[@]}"; do
    AUDIT_ARGS+=(--audit-root "${audit_dir}")
  done

  mkdir -p "$(dirname "${DATASET_OUTPUT_ROOT}")"
  echo "Validating all audit JSON/NPZ pairs and checking quadrant coverage..."
  python -m toolkits.libero_safety.build_dataset \
    "${AUDIT_ARGS[@]}" \
    --output-dir "${DATASET_OUTPUT_ROOT}" \
    --quadrants Q1 Q2 Q3 Q4 \
    --split-unit root \
    --train-fraction 0.8 \
    --val-fraction 0.1 \
    --seed "${EVAL_SEED}" \
    --min-per-quadrant Q1=1 \
    --min-per-quadrant Q2=1 \
    --min-per-quadrant Q3=1 \
    --min-per-quadrant Q4=1 \
    --dry-run

  python -m toolkits.libero_safety.build_dataset \
    "${AUDIT_ARGS[@]}" \
    --output-dir "${DATASET_OUTPUT_ROOT}" \
    --quadrants Q1 Q2 Q3 Q4 \
    --split-unit root \
    --train-fraction 0.8 \
    --val-fraction 0.1 \
    --seed "${EVAL_SEED}" \
    --min-per-quadrant Q1=1 \
    --min-per-quadrant Q2=1 \
    --min-per-quadrant Q3=1 \
    --min-per-quadrant Q4=1

  echo "Dataset summary:"
  cat "${DATASET_OUTPUT_ROOT}/summary.json"
fi

echo "Collection completed successfully."
