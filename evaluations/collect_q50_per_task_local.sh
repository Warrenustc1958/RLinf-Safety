#!/usr/bin/env bash
set -Eeuo pipefail

# Collect at least TARGET_PER_Q valid Q1/Q2/Q3/Q4 episodes for every
# LIBERO-Safety task.  Multiple tasks are evaluated in one RLinf job so the
# two rollout workers can use larger inference batches on H200 GPUs.

export RLINF_ROOT="${RLINF_ROOT:-/openbayes/input/input2/RLinf-Safety}"
export LIBERO_SAFETY_PATH="${LIBERO_SAFETY_PATH:-/openbayes/input/input2/LIBERO-Safety}"
export RUNTIME_ROOT="${RUNTIME_ROOT:-/openbayes/home/rlinf-safety-runtime}"
export VENV_PATH="${VENV_PATH:-${RUNTIME_ROOT}/.venv-openpi-libero-safety}"
export MODEL="${MODEL:-${RUNTIME_ROOT}/checkpoints/pi05_libero_safety_rlinf}"

source "${VENV_PATH}/bin/activate"
cd "${RLINF_ROOT}"

RLINF_LIBERO_RUN_LOCK_PATH="${RLINF_LIBERO_RUN_LOCK_PATH:-/tmp/rlinf-libero-run.lock}"
exec 9>"${RLINF_LIBERO_RUN_LOCK_PATH}"
if ! flock -n 9; then
  echo "Another RLinf/LIBERO job owns ${RLINF_LIBERO_RUN_LOCK_PATH}." >&2
  echo "Do not run natural rollout and Q50 collection at the same time." >&2
  exit 75
fi
export RLINF_LIBERO_RUN_LOCK_HELD=1

export LIBERO_TYPE=safety
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${VENV_PATH}/share/libero-safety}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EMBODIED_PATH="${RLINF_ROOT}/examples/embodiment"
export PYTHONPATH="${LIBERO_SAFETY_PATH}:${RLINF_ROOT}:${PYTHONPATH:-}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export GPU_RANKS="${GPU_RANKS:-0-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Counterfactual epochs can legitimately remain in one collective phase for
# longer than RLinf's 180-minute default, especially for horizon failures.
export RLINF_TIMEOUT="${RLINF_TIMEOUT:-720}"

# Serialize MuJoCo rendering only within a physical GPU.  GPU 0 and GPU 1 can
# render concurrently, while subprocesses sharing one EGL device remain safe.
export RLINF_LIBERO_EGL_LOCK_PATH="${RLINF_LIBERO_EGL_LOCK_PATH:-/tmp/rlinf-libero-egl.lock}"
export RLINF_LIBERO_EGL_LOCK_SCOPE="${RLINF_LIBERO_EGL_LOCK_SCOPE:-per_gpu}"
export RLINF_LIBERO_DEBUG="${RLINF_LIBERO_DEBUG:-0}"

unset RAY_ADDRESS
RUN_TOKEN="${RUN_TOKEN:-$(date +%Y%m%d_%H%M%S)}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/rq-${RUN_TOKEN}}"
export TMPDIR="${TMPDIR:-/tmp/pq-${RUN_TOKEN}}"
mkdir -p "${RAY_TMPDIR}" "${TMPDIR}"

TARGET_PER_Q="${TARGET_PER_Q:-50}"
ROOTS_PER_TASK_PER_ROUND="${ROOTS_PER_TASK_PER_ROUND:-50}"
TASKS_PER_JOB="${TASKS_PER_JOB:-1}"
MAX_ROUNDS="${MAX_ROUNDS:-20}"
MAX_BRANCHES="${MAX_BRANCHES:-8}"
RISK_WINDOW="${RISK_WINDOW:-32}"
MAX_ROLLOUT_STEPS=$(((MAX_BRANCHES + 1) * 512))

if (( TASKS_PER_JOB < 1 )); then
  echo "TASKS_PER_JOB must be at least 1" >&2
  exit 2
fi
if (( ROOTS_PER_TASK_PER_ROUND < 1 || ROOTS_PER_TASK_PER_ROUND > 50 )); then
  echo "ROOTS_PER_TASK_PER_ROUND must be in [1, 50]" >&2
  exit 2
fi

export COLLECT_ROOT="${COLLECT_ROOT:-${RUNTIME_ROOT}/rollouts/pi05_libero_safety_q50/${RUN_TOKEN}}"
mkdir -p "${COLLECT_ROOT}/runs"

coverage() {
  python - "${COLLECT_ROOT}" "$1" "$2" "$3" "${TARGET_PER_Q}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
want_suite = sys.argv[2]
want_level = sys.argv[3]
want_task = int(sys.argv[4])
target = int(sys.argv[5])
counts = Counter()
invalid = 0

for path in root.rglob("*.json"):
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if record.get("schema_version") != "libero_safety_episode_audit/v1":
        continue
    episode = record.get("episode", {})
    if (
        str(episode.get("task_suite")) != want_suite
        or str(episode.get("safety_level")) != want_level
        or int(episode.get("task_id", -1)) != want_task
    ):
        continue
    trajectory = record.get("trajectory", [])
    visual = record.get("visual_trajectory", {})
    relative_path = visual.get("path")
    if (
        not trajectory
        or not relative_path
        or int(visual.get("num_frames", -1)) != len(trajectory) + 1
        or not (path.parent / relative_path).is_file()
    ):
        invalid += 1
        continue
    quadrant = record.get("outcome", {}).get("quadrant", record.get("quadrant"))
    if quadrant in {"Q1", "Q2", "Q3", "Q4"}:
        counts[quadrant] += 1

values = [counts[q] for q in ("Q1", "Q2", "Q3", "Q4")]
print(
    f"[coverage] {want_suite}/{want_level}/task-{want_task}: "
    f"Q1={values[0]} Q2={values[1]} Q3={values[2]} Q4={values[3]} "
    f"invalid={invalid} target={target}"
)
raise SystemExit(0 if all(value >= target for value in values) else 1)
PY
}

task_ids_for_level() {
  python - "$1" "$2" <<'PY' | sed -n 's/^TASK_ID=//p'
import sys
from libero.libero.benchmark import get_benchmark

suite = get_benchmark(sys.argv[1])()
level = int(sys.argv[2])
for task_id in range(suite.get_num_tasks()):
    if int(suite.get_task(task_id).level) == level:
        print(f"TASK_ID={task_id}")
PY
}

SUITES=(affordance human_safety obstacle_avoidance obstacle_avoidance_human)
incomplete_any=0

for suite_index in "${!SUITES[@]}"; do
  suite="${SUITES[$suite_index]}"
  for level_number in 0 1 2; do
    level="L${level_number}"
    mapfile -t all_task_ids < <(task_ids_for_level "${suite}" "${level_number}")
    if (( ${#all_task_ids[@]} == 0 )); then
      echo "No tasks found for ${suite}/${level}" >&2
      exit 2
    fi

    for ((round=1; round<=MAX_ROUNDS; round++)); do
      pending=()
      for task_id in "${all_task_ids[@]}"; do
        if ! coverage "${suite}" "${level}" "${task_id}"; then
          pending+=("${task_id}")
        fi
      done
      if (( ${#pending[@]} == 0 )); then
        break
      fi

      for ((offset=0; offset<${#pending[@]}; offset+=TASKS_PER_JOB)); do
        batch=("${pending[@]:offset:TASKS_PER_JOB}")
        filter="[$(IFS=,; echo "${batch[*]}")]"
        total_envs=$((ROOTS_PER_TASK_PER_ROUND * ${#batch[@]}))
        seed=$((suite_index * 100000 + level_number * 10000 + round * 100 + offset))
        task_tag="$(IFS=-; echo "${batch[*]}")"
        stamp="$(date +%Y%m%d_%H%M%S)"
        round_root="${COLLECT_ROOT}/runs/${suite}_${level}_tasks${task_tag}_round${round}_${stamp}"
        mkdir -p "${round_root}"

        echo "============================================================"
        echo "suite=${suite} level=${level} tasks=${filter} round=${round}"
        echo "total_envs=${total_envs}; approximately $((total_envs / 2)) envs/GPU"
        echo "output=${round_root}"
        echo "============================================================"

        ray stop --force || true
        MODE=cell \
        SUITE="${suite}" \
        LEVEL="${level}" \
        GPU_RANKS="${GPU_RANKS}" \
        TOTAL_ENVS="${total_envs}" \
        EVAL_SEED="${seed}" \
        OUTPUT_ROOT="${round_root}" \
          bash evaluations/run_safelibero_pi05.sh \
            "env.eval.task_id_filter=${filter}" \
            "rollout.model.num_action_chunks=1" \
            "env.eval.max_episode_steps=512" \
            "env.eval.max_steps_per_rollout_epoch=${MAX_ROLLOUT_STEPS}" \
            "env.eval.episode_auditor.enabled=true" \
            "env.eval.episode_auditor.mode=shadow_collect" \
            "env.eval.episode_auditor.include_observations=true" \
            "env.eval.episode_auditor.save_visual_observations=true" \
            "env.eval.episode_auditor.visual_camera_keys=[agentview_image]" \
            "env.eval.episode_auditor.counterfactual_branch.enabled=true" \
            "env.eval.episode_auditor.counterfactual_branch.risk_window_steps=${RISK_WINDOW}" \
            "env.eval.episode_auditor.counterfactual_branch.max_branches_per_root=${MAX_BRANCHES}" \
            "env.eval.episode_auditor.counterfactual_branch.target_quadrants=[Q2,Q4]" \
            "env.eval.episode_auditor.counterfactual_branch.perturbation_range=[0.15,1.0]" \
            "env.eval.episode_auditor.counterfactual_branch.action_dimensions=[0,1,2,3,4,5]" \
          2>&1 | tee "${round_root}/launcher.log"
      done
    done

    for task_id in "${all_task_ids[@]}"; do
      if ! coverage "${suite}" "${level}" "${task_id}"; then
        incomplete_any=1
      fi
    done
  done
done

ray stop --force || true
echo "COLLECT_ROOT=${COLLECT_ROOT}"
if (( incomplete_any != 0 )); then
  echo "Some tasks did not reach all four quadrant targets." >&2
  exit 3
fi
echo "All tasks reached Q1/Q2/Q3/Q4 >= ${TARGET_PER_Q}."
