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

"""P5: does a fresh MuJoCo reset to a trial's init state reproduce the audit's frame 0?

For held-out roots, the env is reset with ``set_init_state(init_states[trial_id])``
(no qpos override) and the agentview render (rotated to policy view) is compared
with audit frame 0, the frame a WM rollout starts from. Reported per suite as mean
|diff| on uint8 pixels.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from toolkits.libero_safety.wan_ac_sft.sampler import load_frames, load_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-task", type=int, default=3)
    args = ap.parse_args()

    from libero.libero.envs import OffScreenRenderEnv
    from rlinf.envs.sim.libero.utils import get_benchmark_overridden

    rows = [r for r in load_manifest(args.index, "test") if r["kind"] == "root"]
    by_task = defaultdict(list)
    for r in rows:
        by_task[(r["suite"], r["task"])].append(r)
    results = []
    for (suite_name, task_id), rs in sorted(by_task.items()):
        suite = get_benchmark_overridden(suite_name)()
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task.level, task.level_id)
        env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(task.level, task.level_id),
                                 camera_heights=256, camera_widths=256, ignore_done=True)
        env.seed(0)
        env.reset()
        seen = set()
        for r in rs:
            if r["trial"] in seen or len(seen) >= args.per_task:
                continue
            seen.add(r["trial"])
            env.set_init_state(init_states[r["trial"]])
            inner = env.env
            inner.sim.forward()
            inner._update_observables(force=True)
            img = np.ascontiguousarray(inner._get_observations()["agentview_image"][::-1, ::-1, :3])
            f0 = load_frames(r["agent_segments"], [0], "agentview_image")[0]
            d = np.abs(img.astype(np.int16) - f0.astype(np.int16))
            results.append({"suite": suite_name, "level": r["level"], "task": task_id, "trial": r["trial"],
                            "mean": float(d.mean()), "p99": float(np.percentile(d, 99)), "max": int(d.max())})
            print(json.dumps(results[-1]), flush=True)
        env.close()

    agg = defaultdict(list)
    for x in results:
        agg[x["suite"]].append(x["mean"])
    summary = {k: {"n": len(v), "mean": float(np.mean(v)), "max": float(np.max(v))} for k, v in agg.items()}
    allm = [x["mean"] for x in results]
    summary["all"] = {"n": len(allm), "mean": float(np.mean(allm)), "median": float(np.median(allm)),
                      "max": float(np.max(allm)), "frac_le_3": float(np.mean(np.array(allm) <= 3.0))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": results}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
