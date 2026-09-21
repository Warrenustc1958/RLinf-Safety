#!/bin/bash
set -euo pipefail

cd /cpfs/xlab/xujingbo/safety/RLinf
source .venv-openvlaoft-libero/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=${MODEL:-/cpfs/xlab/xujingbo/safety/checkpoints/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora}
LORA=${MODEL}/lora_adapter
GPU_RANKS=${GPU_RANKS:-0-3}
TOTAL_ENVS=${TOTAL_ENVS:-128}

run_libero_eval() {
  local config_name="$1"
  local episode_steps="$2"
  local total_steps="$3"

  bash evaluations/run_eval.sh libero "${config_name}" \
    '~cluster.component_placement' \
    "+cluster.component_placement.env=${GPU_RANKS}" \
    "+cluster.component_placement.rollout=${GPU_RANKS}" \
    env.eval.rollout_epoch=1 \
    env.eval.total_num_envs="${TOTAL_ENVS}" \
    env.eval.max_episode_steps="${episode_steps}" \
    env.eval.max_steps_per_rollout_epoch="${total_steps}" \
    env.eval.use_ordered_reset_state_ids=True \
    env.eval.video_cfg.save_video=False \
    rollout.sampling_params.do_sample=False \
    rollout.sampling_params.temperature_eval=-1 \
    rollout.model.model_path=${MODEL} \
    rollout.model.is_lora=True \
    rollout.model.lora_path="${LORA}" \
    rollout.model.unnorm_key=libero_130_no_noops_trajall \
    rollout.enable_offload=False
}
run_libero_eval libero_goal_openvlaoft_eval 512 4096
# Object：预期约 50.20%
#run_libero_eval libero_object_openvlaoft_eval 512 4096

# Spatial：预期约 51.61%
#run_libero_eval libero_spatial_openvlaoft_eval 512 4096

# LIBERO-10 / Long：预期约 11.90%
#run_libero_eval libero_10_openvlaoft_eval 520 4160
