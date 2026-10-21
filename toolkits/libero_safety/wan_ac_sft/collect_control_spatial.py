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

"""Control experiment data: scripted rollouts in standard LIBERO-Spatial.

No expert demos are available on this host, so each episode executes a scripted
action sequence (piecewise-constant xyz translation, re-drawn every
``--segment`` steps, magnitude ``--mag``; random gripper open/close) from a task
init state, and records the agentview frames (rotated 180 to policy view). The
output is an index in the S0 manifest format, so ``ar_eval.py`` runs on it
unchanged. All rows are split="test", kind="root"; the ``quadrant`` field is a
round-robin placeholder used only for stratified selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def scripted_actions(rng, n, segment, mag):
    acts = np.zeros((n, 7), np.float32)
    for s in range(0, n, segment):
        d = rng.normal(size=3)
        d = d / (np.linalg.norm(d) + 1e-8) * mag
        acts[s:s + segment, :3] = d
        acts[s:s + segment, 6] = rng.choice([-1.0, 1.0])
    return acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--inits-per-task", type=int, default=2)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--segment", type=int, default=16)
    ap.add_argument("--mag", type=float, default=0.6)
    args = ap.parse_args()

    import torch
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from rlinf.envs.sim.libero.utils import get_benchmark_overridden

    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    suite = get_benchmark_overridden("libero_spatial")()
    rows, idx = [], 0
    for task_id in range(suite.get_num_tasks()):
        task = suite.get_task(task_id)
        # LIBERO-Safety's wrapper expects <suite>/L<level>/ subdirs; the standard
        # libero_spatial files are flat, so build the paths directly.
        bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
        init_states = torch.load(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
                                 weights_only=False)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        env.seed(0)
        env.reset()
        for k in range(args.inits_per_task):
            rng = np.random.default_rng(1000 * task_id + k)
            obs = env.set_init_state(init_states[k])
            acts = scripted_actions(rng, args.steps, args.segment, args.mag)
            frames = [obs["agentview_image"][::-1, ::-1]]
            for a in acts:
                obs, _r, _done, _info = env.step(a)
                frames.append(obs["agentview_image"][::-1, ::-1])
            frames = np.ascontiguousarray(np.stack(frames)).astype(np.uint8)
            fpath = out / "frames" / f"{idx:04d}.npz"
            np.savez_compressed(fpath, agentview_image=frames)
            lpath = out / "labels" / f"{idx:04d}.npz"
            np.savez(lpath, actions=acts)
            rows.append({"idx": idx, "json": str(fpath), "episode_id": f"spatial_t{task_id}_i{k}",
                         "suite": "libero_spatial", "level": "-", "cell": "libero_spatial", "task": task_id,
                         "trial": k, "quadrant": ["Q1", "Q2", "Q3", "Q4"][idx % 4], "kind": "root",
                         "num_frames": len(frames), "agent_segments": [[str(fpath), 0, len(frames)]],
                         "wrist_segments": None, "dual": False, "split": "test", "labels": str(lpath),
                         "branch_frame": None, "first_violation_frame": None, "first_success_frame": None,
                         "prefix_available": True, "instruction": task.language})
            print(f"task {task_id} init {k}: {len(frames)} frames", flush=True)
            idx += 1
        env.close()
    with (out / "manifest.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("episodes", len(rows))


if __name__ == "__main__":
    main()
