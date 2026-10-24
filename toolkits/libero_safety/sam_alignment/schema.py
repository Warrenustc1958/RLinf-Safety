# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared constants and the time-axis contract of the SAM alignment pipeline.

Time axis (0-based global action index, ``label_alignment: post_action_state``)::

    frame[t] + action_env[t] -> frame[t + 1]
    cost[t, k] describes the safety state of frame[t + 1]

A chunk starting at t0 (t0 % 8 == 0) has actions / labels t0 .. t0+7 and future
frames t0+1 .. t0+8. Its Wan window is 13 frames: anchor frame 0, memory frames
t0-3 .. t0 (clamped at 0, the sampler's START_TEMPLATE for t0 = 0) and the 8
future frames. Branches are stitched as root[:bt-1] + branch (bt is 1-based; the
on-disk branch frame 0 equals root frame bt-1).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

CHANNELS = ("c_obj", "c_robot", "c_grasp")
K = len(CHANNELS)
CH = {c: i for i, c in enumerate(CHANNELS)}
PROX_CHANNELS = (0, 1)  # proximity aux only for c_obj / c_robot
CHUNK = 8
WINDOW = 13
DIST_MAX = 0.25
NEAR_MISS_M = 0.02

INDEX_ROOT = Path("/data_cf/wan_ac_sft/index")
INDEXES = ("cws_v1", "cws_test_new")
OUT_ROOT = Path("/data_cf/cws_sam/v1")

# audit predicate -> SAM channel (checkgripperforce is deliberately unsupervised)
PRED_CHANNEL = {"checkcontact": "c_obj", "checkrobotcontact": "c_robot", "wronggrasppart": "c_grasp"}


def uid(index: str, idx: str) -> str:
    return f"{index}/{idx}"


def group_key(row: dict) -> str:
    """Initial-state group: (suite, level, task_id, trial_id)."""
    return f"{row['suite']}|{row['level']}|{row['task']}|{row['trial']}"


def audit_json_of_npz(npz_path: str) -> str:
    d, n = os.path.split(npz_path)
    return os.path.join(os.path.dirname(d), n.replace(".npz", ".json"))


def proximity(dist: np.ndarray) -> np.ndarray:
    """Near-contact aux target clip(1 - d / 0.02, 0, 1); 1 at contact or penetration."""
    return np.clip(1.0 - dist / NEAR_MISS_M, 0.0, 1.0)


def chunk_starts(num_frames: int) -> list[int]:
    """Chunk starts t0 = 0, 8, ... with at least one real future frame."""
    return list(range(0, num_frames - 1, CHUNK))


def chunk_window(t0: int, num_frames: int) -> tuple[list[int], int]:
    """13 frame ids (anchor, 4 memory, 8 future; tail padded with the last frame) and valid length."""
    last = num_frames - 1
    memory = [max(0, f) for f in range(t0 - 3, t0 + 1)]
    future = [min(f, last) for f in range(t0 + 1, t0 + CHUNK + 1)]
    return [0] + memory + future, min(CHUNK, last - t0)
