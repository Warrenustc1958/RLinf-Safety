#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$SCRIPT_DIR")"
WORKSPACE_ROOT="$(dirname "$REPO_PATH")"
VENV_PATH="${VENV_PATH:-${WORKSPACE_ROOT}/.venv-openvlaoft-libero-safety}"
# ========== 更新模型权重路径 global_step_2000 ==========
ACTOR_CKPT="${ACTOR_CKPT:-/oss/xujingbo/checkpoints/rlinf/libero_safety_goal_traj1_init_full_sft_2000/openvlaoft_goal_traj1_init_full_2000/checkpoints/global_step_2000/actor}"
TRAIN_CONFIG="${TRAIN_CONFIG:-/oss/xujingbo/checkpoints/rlinf/libero_safety_goal_traj1_init_full_sft_2000/tensorboard/config.yaml}"
HF_MODEL="${HF_MODEL:-${WORKSPACE_ROOT}/checkpoints/libero_safety_goal_traj1_init_full_step2000_hf}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/oss/xujingbo/evaluations/libero_safety/openvlaoft_goal_traj1_init_full_2000_step2000}"
DATASET_STATS="${DATASET_STATS:-${WORKSPACE_ROOT}/datasets/libero_safety_one_shot/meta/stats.json}"
GPU_RANKS="${GPU_RANKS:-0-4}"
TOTAL_ENVS="${TOTAL_ENVS:-50}"
EVAL_SEED="${EVAL_SEED:-0}"

source "${VENV_PATH}/bin/activate"
cd "$REPO_PATH"
export REPO_PATH
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

mkdir -p "$HF_MODEL" "$OUTPUT_ROOT"
if [ ! -f "${HF_MODEL}/.conversion_complete" ]; then
  python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf \
    +convertor.train_config_path="$TRAIN_CONFIG" \
    convertor.ckpt_path="${ACTOR_CKPT}/model_state_dict/full_weights.pt" \
    convertor.save_path="$HF_MODEL" \
    convertor.torch_dtype=bf16
  touch "${HF_MODEL}/.conversion_complete"
fi

suites=(affordance human_safety obstacle_avoidance obstacle_avoidance_human)
levels=(L0 L1 L2)
for suite in "${suites[@]}"; do
  for level in "${levels[@]}"; do
    run_name="${suite}_${level}"
    run_dir="${OUTPUT_ROOT}/${run_name}"
    mkdir -p "$run_dir"
    echo "Running LIBERO-Safety main-table cell: ${run_name}"
    env \
      MODE=eval \
      SUITE="$suite" \
      LEVEL="$level" \
      GPU_RANKS="$GPU_RANKS" \
      TOTAL_ENVS="$TOTAL_ENVS" \
      MODEL="$HF_MODEL" \
      UNNORM_KEY=libero_safety \
      IS_LORA=False \
      RLINF_RUNTIME_DIR="/r-eval-${suite}-${level}" \
      bash evaluations/run_safelibero.sh \
        +rollout.model.dataset_statistics_path="$DATASET_STATS" \
        env.eval.seed="$EVAL_SEED" \
        env.eval.ignore_terminations=False \
        runner.logger.log_path="$run_dir" \
        runner.logger.experiment_name="openvlaoft_goal_traj1_init_full_2000_step2000_${run_name}"
  done
done
