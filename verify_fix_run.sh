#!/bin/bash
cd /cpfs/xlab/xujingbo/safety/RLinf
export MODE=cell
export SUITE=obstacle_avoidance
export LEVEL=L0
export GPU_RANKS=0
export TOTAL_ENVS=10
export EVAL_SEED=0
export OUTPUT_ROOT=/cpfs/xlab/xujingbo/safety/datasets/libero_safety_wan_v1/pilot_branch_verify

bash evaluations/run_safelibero_pi05.sh   env.eval.episode_auditor.enabled=true   env.eval.episode_auditor.mode=shadow_collect   env.eval.episode_auditor.save_visual_observations=true   env.eval.episode_auditor.visual_camera_keys='[agentview_image]'   env.eval.episode_auditor.counterfactual_branch.enabled=true   env.eval.episode_auditor.counterfactual_branch.max_branches_per_root=4   env.eval.max_steps_per_rollout_epoch=2560   env.eval.video_cfg.save_video=false
