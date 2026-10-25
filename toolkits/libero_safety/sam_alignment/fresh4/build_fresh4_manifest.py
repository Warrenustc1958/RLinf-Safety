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

"""fresh4 step 7: build a cws-style manifest.jsonl from the finished collection cells.

Row format matches /data_cf/wan_ac_sft/index/cws_v1/manifest.jsonl as far as
build_labels.process() needs it. Frame segments follow the cws convention:
  root    [[root_npz, 0, T+1]]                               num_frames = T + 1
  branch  [[root_npz, 0, bt-1], [branch_npz, 0, Tb+1]]       num_frames = bt - 1 + Tb + 1
(branch frame 0 = root frame bt-1). trial = 1000 + fresh trial_id, so the group key
suite|level|task|trial never collides with shipped-state groups (trial 0-49);
role comes from split_fresh4.json (trial_id // 10).
Admission: root audit present, stitched length <= 512, both camera views stored.

Then label with the grader:
  python -m toolkits.libero_safety.sam_alignment.build_labels \
      --index_root <fresh4>/index --indexes fresh4 --out <fresh4>/labels --workers 16
"""

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

F = Path("/data_cf/cws_sam/v1/fresh4")
RUNS = Path("/data_cf/cws/fresh4_cf/runs")


def vis_of(js: Path) -> Path:
    return js.parent / "visual" / js.name.replace(".json", ".npz")


def has_views(npz: Path) -> bool:
    try:
        names = set(zipfile.ZipFile(npz).namelist())
    except Exception:
        return False
    return {"agentview_image.npy", "robot0_eye_in_hand_image.npy"} <= names


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(F / "index" / "fresh4" / "manifest.jsonl"))
    args = ap.parse_args()
    split = json.load(open(F / "split_fresh4.json"))
    roles = {int(k): v for k, v in split["trial_role"].items()}
    rows, skip, n = [], Counter(), 0
    for cell_dir in sorted(RUNS.iterdir()):
        log = cell_dir / "launcher.log"
        if not log.exists() or "EXIT=0" not in log.read_text(errors="ignore"):
            skip["cell_unfinished"] += 1
            continue
        recs = {}
        for js in cell_dir.rglob("process-*.json"):
            if "visual" in js.parts:
                continue
            recs[js] = json.load(open(js))
        by_id = {(js.parent, r["episode"]["episode_id"]): js for js, r in recs.items()}
        for js, r in sorted(recs.items()):
            e = r["episode"]
            T = len(r["trajectory"])
            suite, level, task, trial = e["task_suite"], e["safety_level"], int(e["task_id"]), int(e["trial_id"])
            base = {"src": "fresh4", "cell": f"{suite}_{level}", "tcell": f"{suite}_{level}_t{task}",
                    "suite": suite, "level": level, "task": task, "trial": 1000 + trial, "trial_fresh": trial,
                    "role": roles[trial], "split": roles[trial], "json": str(js), "dual": True,
                    "quadrant": r.get("quadrant"), "quadrant_collector": r.get("quadrant"),
                    "program_id": e.get("program_id") or f"__root__:{suite}_{level}_t{task}",
                    "program_params": e.get("program_params")}
            if not r.get("root_episode_id"):
                npz = vis_of(js)
                if not has_views(npz):
                    skip["no_views"] += 1
                    continue
                row = dict(base, kind="root", branch_timestep=None, branch_frame=None, num_frames=T + 1,
                           agent_segments=[[str(npz), 0, T + 1]], wrist_segments=[[str(npz), 0, T + 1]])
            else:
                root_js = by_id.get((js.parent, r["root_episode_id"]))
                if root_js is None:
                    skip["no_root"] += 1
                    continue
                bt = int(e["branch_timestep"])
                if bt - 1 + T > 512:
                    skip["over_512"] += 1
                    continue
                rn, bn = vis_of(root_js), vis_of(js)
                if not (has_views(rn) and has_views(bn)):
                    skip["no_views"] += 1
                    continue
                segs = [[str(rn), 0, bt - 1], [str(bn), 0, T + 1]]
                row = dict(base, kind="counterfactual_branch", branch_timestep=bt, branch_frame=bt,
                           num_frames=bt - 1 + T + 1, agent_segments=segs, wrist_segments=segs,
                           root_episode_json=str(root_js))
            row["idx"] = f"f{n:05d}"
            n += 1
            rows.append(row)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    c = Counter((r["role"], r["kind"]) for r in rows)
    print(f"{len(rows)} rows -> {out}; skipped {dict(skip)}")
    for k in sorted(c):
        print(" ", k, c[k])


if __name__ == "__main__":
    main()
