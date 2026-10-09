#!/usr/bin/env bash
set -Eeuo pipefail

# A bounded, one-chunk probe for locating LIBERO-Safety evaluation stalls.
#
#   PROBE=single  -> 1 GPU, 1 env   (basic model/simulator/collective path)
#   PROBE=dual    -> 2 GPUs, 2 envs (cross-rank collective path)
#   PROBE=dense   -> 2 GPUs, 50 envs (production reset/EGL-lock pressure)
#
# The script owns the host Ray runtime. Do not run it beside another RLinf job.

RLINF_ROOT="${RLINF_ROOT:-/openbayes/input/input2/RLinf-Safety}"
LIBERO_SAFETY_PATH="${LIBERO_SAFETY_PATH:-/openbayes/input/input2/LIBERO-Safety}"
VENV_PATH="${VENV_PATH:-/openbayes/home/rlinf-safety-runtime/.venv-openpi-libero-safety}"
MODEL="${MODEL:-/openbayes/home/rlinf-safety-runtime/checkpoints/pi05_libero_safety_rlinf}"
PROBE="${PROBE:-single}"
SUITE="${SUITE:-affordance}"
LEVEL="${LEVEL:-L0}"
TASK_ID="${TASK_ID:-0}"
RESET_WAIT="${RESET_WAIT:-1}"
DEBUG_TIMEOUT="${DEBUG_TIMEOUT:-900}"
STACK_INTERVAL="${STACK_INTERVAL:-120}"

case "$PROBE" in
  single)
    GPU_RANKS="${GPU_RANKS:-0}"
    TOTAL_ENVS="${TOTAL_ENVS:-1}"
    ;;
  dual)
    GPU_RANKS="${GPU_RANKS:-0-1}"
    TOTAL_ENVS="${TOTAL_ENVS:-2}"
    ;;
  dense)
    GPU_RANKS="${GPU_RANKS:-0-1}"
    TOTAL_ENVS="${TOTAL_ENVS:-50}"
    ;;
  *)
    echo "PROBE must be single, dual, or dense; got: $PROBE" >&2
    exit 2
    ;;
esac

for required in \
  "$RLINF_ROOT/evaluations/run_safelibero_pi05.sh" \
  "$LIBERO_SAFETY_PATH/libero/libero/__init__.py" \
  "$VENV_PATH/bin/activate" \
  "$MODEL/model.safetensors"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

RUN_LOCK="${RLINF_LIBERO_RUN_LOCK_PATH:-/tmp/rlinf-libero-run.lock}"
exec 9>"$RUN_LOCK"
if ! flock -n 9; then
  echo "Another RLinf/LIBERO job owns $RUN_LOCK; stop it before debugging." >&2
  exit 75
fi
export RLINF_LIBERO_RUN_LOCK_HELD=1

timestamp="$(date +%Y%m%d-%H%M%S)"
DIAG_ROOT="${DIAG_ROOT:-/openbayes/home/rlinf-safety-runtime/debug/pi05-libero-${PROBE}-${timestamp}}"
export OUTPUT_ROOT="$DIAG_ROOT/output"
export RAY_TMPDIR="/tmp/rd-${PROBE}-${timestamp}"
export TMPDIR="/tmp/pd-${PROBE}-${timestamp}"
mkdir -p "$DIAG_ROOT" "$OUTPUT_ROOT" "$RAY_TMPDIR" "$TMPDIR"
exec > >(tee -a "$DIAG_ROOT/debug.log") 2>&1

source "$VENV_PATH/bin/activate"
cd "$RLINF_ROOT"

export LIBERO_TYPE=safety
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$VENV_PATH/share/libero-safety}"
export PYTHONPATH="$LIBERO_SAFETY_PATH:$RLINF_ROOT:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UV_LINK_MODE=copy
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export RAY_DEDUP_LOGS=0
export RLINF_TIMEOUT="${RLINF_TIMEOUT:-30}"
export RLINF_EVAL_DEBUG=1
export RLINF_LIBERO_DEBUG=1
export RLINF_DEBUG_STACK_INTERVAL="$STACK_INTERVAL"
export RLINF_LIBERO_EGL_LOCK_PATH="${RLINF_LIBERO_EGL_LOCK_PATH:-/tmp/rlinf-libero-egl.lock}"
export RLINF_LIBERO_EGL_LOCK_SCOPE="${RLINF_LIBERO_EGL_LOCK_SCOPE:-per_gpu}"

{
  echo "timestamp=$timestamp"
  echo "probe=$PROBE gpu_ranks=$GPU_RANKS total_envs=$TOTAL_ENVS"
  echo "suite=$SUITE level=$LEVEL task_id=$TASK_ID reset_wait=$RESET_WAIT"
  echo "rlinf_root=$RLINF_ROOT"
  echo "libero_safety_path=$LIBERO_SAFETY_PATH"
  echo "libero_config_path=$LIBERO_CONFIG_PATH"
  echo "venv_path=$VENV_PATH"
  echo "model=$MODEL"
  echo "ray_tmpdir=$RAY_TMPDIR"
  echo "egl_lock=$RLINF_LIBERO_EGL_LOCK_PATH scope=$RLINF_LIBERO_EGL_LOCK_SCOPE"
  git rev-parse HEAD
  git status --short
  python -V
  which python
  uname -a
} | tee "$DIAG_ROOT/preflight.txt"

nvidia-smi -L | tee "$DIAG_ROOT/nvidia-smi-L.txt" || true
nvidia-smi | tee "$DIAG_ROOT/nvidia-smi-before.txt" || true

python - <<'PY' | tee "$DIAG_ROOT/python-env.txt"
import os
import numpy
import torch
import libero.libero as libero

print("numpy:", numpy.__version__)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda count:", torch.cuda.device_count())
print("libero module:", libero.__file__)
for key in ("benchmark_root", "bddl_files", "init_states", "assets", "datasets"):
    try:
        print(f"libero {key}:", libero.get_libero_path(key))
    except Exception as exc:
        print(f"libero {key}: ERROR {type(exc).__name__}: {exc}")
for key in (
    "LIBERO_TYPE", "LIBERO_CONFIG_PATH", "MUJOCO_GL", "PYOPENGL_PLATFORM",
    "CUDA_VISIBLE_DEVICES", "RLINF_LIBERO_EGL_LOCK_PATH",
    "RLINF_LIBERO_EGL_LOCK_SCOPE",
):
    print(f"env {key}:", os.environ.get(key))
PY

# The lock above guarantees that this cannot terminate another managed RLinf run.
ray stop --force || true

monitor_system() {
  while true; do
    date --iso-8601=seconds
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory \
      --format=csv,noheader 2>&1 || true
    ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd --sort=-pcpu \
      | grep -E 'ray|EnvGroup|RolloutGroup|eval_embodied|spawn_main' \
      | grep -v grep | head -n 100 || true
    sleep 30
  done
}

monitor_system >"$DIAG_ROOT/system-samples.log" 2>&1 &
monitor_pid=$!
cleanup_monitor() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup_monitor EXIT

set +e
timeout --signal=INT --kill-after=60s "$DEBUG_TIMEOUT" \
  env \
    MODE=cell \
    SUITE="$SUITE" \
    LEVEL="$LEVEL" \
    GPU_RANKS="$GPU_RANKS" \
    TOTAL_ENVS="$TOTAL_ENVS" \
    MODEL="$MODEL" \
    VENV_PATH="$VENV_PATH" \
    LIBERO_SAFETY_PATH="$LIBERO_SAFETY_PATH" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
  bash evaluations/run_safelibero_pi05.sh \
    "env.eval.task_id_filter=[$TASK_ID]" \
    "env.eval.max_episode_steps=5" \
    "env.eval.max_steps_per_rollout_epoch=5" \
    "env.eval.num_steps_wait=$RESET_WAIT" \
    "env.eval.skip_intermediate_renders=true" \
    "env.eval.progress_log_interval=1" \
    "env.eval.video_cfg.save_video=false" \
    "env.eval.episode_auditor.enabled=false" \
    "rollout.model.num_action_chunks=5"
run_status=$?
set -e

cleanup_monitor
trap - EXIT

echo "probe_exit_status=$run_status" | tee "$DIAG_ROOT/result.txt"
ray status >"$DIAG_ROOT/ray-status.txt" 2>&1 || true
nvidia-smi >"$DIAG_ROOT/nvidia-smi-after.txt" 2>&1 || true
ps -ef >"$DIAG_ROOT/ps-after.txt" 2>&1 || true

if [[ -d "$RAY_TMPDIR/session_latest/logs" ]]; then
  tar -czf "$DIAG_ROOT/ray-logs.tgz" \
    -C "$RAY_TMPDIR/session_latest" logs 2>/dev/null || true
fi

grep -E \
  'eval-debug|eval-progress|libero-worker|Timeout|Traceback|ERROR|Error|actor is dead|CUDA|NCCL|Gloo' \
  "$DIAG_ROOT/debug.log" >"$DIAG_ROOT/key-events.log" || true

ray stop --force || true
echo "Diagnostic bundle: $DIAG_ROOT"
echo "Return debug.log, key-events.log, system-samples.log, preflight.txt, and ray-logs.tgz."
exit "$run_status"
