# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""T1: per-frame EEF pose conditioning for the dual-view Wan (TI2V2, five-frame condition).

Every frame's proprio (eef pos, quat, gripper qpos) is encoded as 11 features (normalised pos, 6-D rotation =
first two columns of the rotation matrix, normalised gripper qpos) and appended after the 7 action dims of the
action tensor, so the window becomes [B, 13, 7 + 11] and the trainer / pipeline plumbing is unchanged.
In the DiT:
  action_mlp1 (cross-attn tokens) sees only the 7 action dims, as before;
  action_mlp2 (per latent frame modulation, 4 frames x D grouped) sees only the action dims, as before, plus
  one zero-initialised head per view maps the grouped pose (4 x 11) to the same modulation; the agent head's
  output is added to the agent half of the dual tokens, the wrist head's to the wrist half
  (patch_action_tokens repeats each latent frame's modulation once per view, agent first).
Zero-initialised output layers: at step 0 the model is exactly dual-41k. Training uses the measured state,
inference a causal estimate (action integration from the start state, WAN_DRIFT_PLAN R4).
"""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn

ACT_DIM = 7
POSE_FEAT = 11


def pose_features(st: np.ndarray, stats: dict) -> np.ndarray:
    """[T, 9] (pos, quat xyzw, gripper qpos) -> [T, 11] float32."""
    from scipy.spatial.transform import Rotation as Rot

    mu, sd = np.asarray(stats["mean"], np.float32), np.asarray(stats["std"], np.float32)
    pos = (st[:, :3] - mu[:3]) / sd[:3]
    m = Rot.from_quat(st[:, 3:7]).as_matrix()
    rot6 = np.concatenate([m[:, :, 0], m[:, :, 1]], 1)
    grip = (st[:, 7:9] - mu[7:9]) / np.maximum(sd[7:9], 1e-6)
    return np.concatenate([pos, rot6, grip], 1).astype(np.float32)


def load_stats(path: str) -> dict:
    return json.load(open(path))


def _head(dim: int) -> nn.Sequential:
    h = nn.Sequential(nn.Linear(4 * POSE_FEAT, dim), nn.SiLU(), nn.Linear(dim, dim))
    nn.init.zeros_(h[2].weight)
    nn.init.zeros_(h[2].bias)
    return h


def enable_pose(dit: nn.Module, views: int = 2) -> None:
    """Add the pose heads to ``dit`` (registered submodules: trained, saved) and route the extended action
    tensor. Call instead of / after ``dual_view.patch_action_tokens``; replaces action_mlp2's forward."""
    if getattr(dit, "_pose_enabled", False):
        return
    dim = dit.action_mlp2[-1].out_features
    dev = dit.action_mlp2[-1].weight.device
    dit.pose_mlp_agent = _head(dim).to(dev)
    dit.pose_mlp_wrist = _head(dim).to(dev)
    m1, m2 = dit.action_mlp1, dit.action_mlp2
    seq1, seq2 = type(m1).forward, type(m2).forward

    def fwd1(x, _m=m1):
        return seq1(_m, x[..., :ACT_DIM] if x.shape[-1] > ACT_DIM else x)

    def fwd2(x, _m=m2):
        d = x.shape[-1] // 4
        if d == ACT_DIM:  # no pose in the window: dual-41k behaviour
            y = seq2(_m, x)
            return y.repeat_interleave(views, dim=-2) if views > 1 else y
        g = x.reshape(*x.shape[:-1], 4, d)
        y = seq2(_m, g[..., :ACT_DIM].flatten(-2))
        pose = g[..., ACT_DIM:].flatten(-2).to(dit.pose_mlp_agent[0].weight.dtype)
        pa = dit.pose_mlp_agent(pose).to(y.dtype)
        if views == 1:
            return y + pa
        pw = dit.pose_mlp_wrist(pose).to(y.dtype)
        y = y.repeat_interleave(views, dim=-2)
        add = torch.stack([pa, pw], dim=-2).flatten(-3, -2)  # agent, wrist per latent frame
        return y + add

    m1.forward, m2.forward = fwd1, fwd2
    m2._views = views
    dit._pose_enabled = True


def pose_state_dict(dit: nn.Module) -> dict:
    return {k: v for k, v in dit.state_dict().items() if k.startswith(("pose_mlp_agent", "pose_mlp_wrist"))}


def load_pose_heads(dit: nn.Module, path: str) -> None:
    """Load pose heads from a T1 export (safetensors with the full DiT, or a pose-only file)."""
    from safetensors.torch import load_file

    sd = {k: v for k, v in load_file(path).items() if k.startswith(("pose_mlp_agent", "pose_mlp_wrist"))}
    missing = dit.load_state_dict(sd, strict=False)
    assert not [k for k in missing.missing_keys if k.startswith("pose_mlp")], missing.missing_keys
