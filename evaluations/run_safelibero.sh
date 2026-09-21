#!/bin/bash
# SafeLIBERO eval（参考 run_libero.sh 的格式）
cd /cpfs/xlab/xujingbo/safety/RLinf
source .venv-openvaloft-safelibero/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LIBERO_TYPE=safe
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# /tmp is nearly full, while CPFS cannot host Ray's Unix sockets.  /dev/shm is
# local, socket-capable, and has ample capacity on this machine.
export RAY_TMPDIR=/dev/shm/rlinf-safelibero-ray
mkdir -p "${RAY_TMPDIR}"

GPU_RANKS=${GPU_RANKS:-0-3}
SAFETY_LEVEL=${SAFETY_LEVEL:-I}
TOTAL_ENVS=${TOTAL_ENVS:-40}
SUITE=${SUITE:-goal}

case "${SUITE}" in
  goal)
    CONFIG_NAME=safelibero_goal_openvlaoft_eval
    DEFAULT_MODEL=/cpfs/xlab/xujingbo/safety/checkpoints/Openvla-oft-SFT-libero-goal-traj1
    DEFAULT_UNNORM_KEY=libero_goal_no_noops
    DEFAULT_IS_LORA=False
    ;;
  object)
    CONFIG_NAME=safelibero_object_openvlaoft_eval
    DEFAULT_MODEL=/cpfs/xlab/xujingbo/safety/checkpoints/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora
    DEFAULT_UNNORM_KEY=libero_130_no_noops_trajall
    DEFAULT_IS_LORA=True
    ;;
  spatial)
    CONFIG_NAME=safelibero_spatial_openvlaoft_eval
    DEFAULT_MODEL=/cpfs/xlab/xujingbo/safety/checkpoints/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora
    DEFAULT_UNNORM_KEY=libero_130_no_noops_trajall
    DEFAULT_IS_LORA=True
    ;;
  long|libero_10)
    CONFIG_NAME=safelibero_10_openvlaoft_eval
    DEFAULT_MODEL=/cpfs/xlab/xujingbo/safety/checkpoints/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora
    DEFAULT_UNNORM_KEY=libero_130_no_noops_trajall
    DEFAULT_IS_LORA=True
    ;;
  *)
    echo "Unknown SUITE=${SUITE}; choose goal, object, spatial, long, or libero_10" >&2
    exit 2
    ;;
esac

MODEL=${MODEL:-${DEFAULT_MODEL}}
UNNORM_KEY=${UNNORM_KEY:-${DEFAULT_UNNORM_KEY}}
IS_LORA=${IS_LORA:-${DEFAULT_IS_LORA}}
LORA_PATH=${LORA_PATH:-${MODEL}/lora_adapter}

run_safelibero_eval() {
  local config_name="$1"
  local total_envs="$2"
  local args=(
    libero "${config_name}"
    '~cluster.component_placement'
    "+cluster.component_placement.env=${GPU_RANKS}"
    "+cluster.component_placement.rollout=${GPU_RANKS}"
    "env.eval.safety_level=${SAFETY_LEVEL}"
    env.eval.rollout_epoch=1
    "env.eval.total_num_envs=${total_envs}"
    env.eval.use_ordered_reset_state_ids=True
    env.eval.video_cfg.save_video=False
    rollout.sampling_params.do_sample=False
    rollout.sampling_params.temperature_eval=-1
    "rollout.model.model_path=${MODEL}"
    "rollout.model.is_lora=${IS_LORA}"
    "rollout.model.unnorm_key=${UNNORM_KEY}"
    rollout.enable_offload=False
  )
  if [[ "${IS_LORA,,}" == "true" ]]; then
    args+=("rollout.model.lora_path=${LORA_PATH}")
  fi
  bash evaluations/run_eval.sh "${args[@]}"
}

# Every SafeLIBERO suite contains four tasks and 50 held-out initial states per
# task.  Use TOTAL_ENVS=40 for a smoke evaluation and 200 for the full split.
run_safelibero_eval "${CONFIG_NAME}" "${TOTAL_ENVS}"
