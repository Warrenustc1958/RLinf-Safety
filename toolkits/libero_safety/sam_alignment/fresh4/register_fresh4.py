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

"""fresh4 step 8: register the four fresh4 roles in splits/split_registry.json.

Adds cost_hard / h_deep / calib_fresh / test_fresh to roles (same stats as
build_splits), usage_history, group_role, episode_role and leakage_checks (fresh4
groups vs every existing group, pairwise between the four roles, branch/root role
mismatch). The previous registry is kept as split_registry.before_fresh4.json.
"""

import collections
import json
import shutil
import time
from pathlib import Path

import numpy as np

CHANNELS = ("c_obj", "c_robot", "c_grasp")
SPLITS = Path("/data_cf/cws_sam/v1/splits")
F = Path("/data_cf/cws_sam/v1/fresh4")
ROLES = ("cost_hard", "h_deep", "calib_fresh", "test_fresh")
USAGE = {
    "cost_hard": "fresh4: hard same-state pairs for base cost re-training (Step 6); also cost training rows",
    "h_deep": "fresh4: deep AR (17-32 chunks) for the underestimation head h (Step 7); never used to fit or select cost",
    "calib_fresh": "fresh4: Step 8 threshold calibration ONLY; never used for training or model selection",
    "test_fresh": "fresh4: final report ONLY, evaluated once after every design choice is frozen; never used for selection",
}


def main():
    reg_path = SPLITS / "split_registry.json"
    reg = json.load(open(reg_path))
    if any(r in reg["roles"] for r in ROLES):
        raise SystemExit("fresh4 roles already registered")
    backup = SPLITS / "split_registry.before_fresh4.json"
    if not backup.exists():
        shutil.copy2(reg_path, backup)
    man = {json.loads(l)["idx"]: json.loads(l) for l in open(F / "index" / "fresh4" / "manifest.jsonl")}
    eps = [json.loads(l) for l in open(F / "labels" / "episodes.jsonl")]
    role_of_ep = {e["uid"]: man[e["idx"]]["role"] for e in eps}
    group_role = {}
    for e in eps:
        r = role_of_ep[e["uid"]]
        if group_role.setdefault(e["group"], r) != r:
            raise SystemExit(f"group {e['group']} in two roles")

    stats = {r: {"groups": set(), "episodes": 0, "roots": 0, "branches": 0} for r in ROLES}
    counts = {r: np.zeros((len(CHANNELS), 3), np.int64) for r in ROLES}
    gwe = {r: collections.defaultdict(set) for r in ROLES}
    for e in eps:
        r = role_of_ep[e["uid"]]
        s = stats[r]
        s["groups"].add(e["group"])
        s["episodes"] += 1
        s["roots" if e["kind"] == "root" else "branches"] += 1
        with np.load(e["labels"]) as z:
            pos = z["cost"] & z["valid_channel"]
        ev = (pos[1:] & ~pos[:-1]).sum(0) + pos[0] if len(pos) else np.zeros(len(CHANNELS))
        counts[r] += np.stack([pos.sum(0), ev, (ev > 0)], 1)
        for c, t in e["first_event_by_channel"].items():
            if t is not None:
                gwe[r][c].add(e["group"])

    old_groups = set(reg["group_role"])
    checks = {}
    for r in ROLES:
        checks[f"{r}&existing_groups"] = len(stats[r]["groups"] & old_groups)
    for i, a in enumerate(ROLES):
        for b in ROLES[i + 1:]:
            checks[f"{a}&{b}"] = len(stats[a]["groups"] & stats[b]["groups"])
    checks["fresh4_branch_root_role_mismatch"] = sum(
        1 for e in eps if e["kind"] != "root" and e.get("root_uid") and role_of_ep[e["root_uid"]] != role_of_ep[e["uid"]])
    checks["fresh4_branch_without_root_uid"] = sum(1 for e in eps if e["kind"] != "root" and not e.get("root_uid"))
    bad = {k: v for k, v in checks.items() if v}
    if bad:
        raise SystemExit(f"leakage / consistency check failed: {bad}")

    for r in ROLES:
        s = stats[r]
        reg["roles"][r] = {"groups": len(s["groups"]), "episodes": s["episodes"], "roots": s["roots"],
                           "branches": s["branches"],
                           "channels": {c: {"pos_steps": int(counts[r][k, 0]), "events": int(counts[r][k, 1]),
                                            "episodes_with_event": int(counts[r][k, 2]),
                                            "groups_with_event": len(gwe[r][c])} for k, c in enumerate(CHANNELS)}}
    reg["usage_history"].update(USAGE)
    reg["leakage_checks"].update(checks)
    reg["group_role"] = dict(sorted({**reg["group_role"], **group_role}.items()))
    reg["episode_role"] = dict(sorted({**reg["episode_role"], **role_of_ep}.items()))
    reg["fresh4"] = {"registered": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "labels_dir": str(F / "labels"), "manifest": str(F / "index" / "fresh4" / "manifest.jsonl"),
                     "split_rule": "fresh trial_id 0-9 cost_hard, 10-19 h_deep, 20-29 calib_fresh, 30-39 test_fresh; "
                                   "group trial = 1000 + fresh trial_id; branches follow root",
                     "init_states_root": "/data_cf/cws/fresh_init/v2 (seed_base 30261025)",
                     "collection": "/data_cf/cws/fresh4_cf/runs"}
    tmp = reg_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=1))
    tmp.replace(reg_path)
    print("checks", checks)
    for r in ROLES:
        print(r, {k: v for k, v in reg["roles"][r].items() if k != "channels"},
              {c: reg["roles"][r]["channels"][c]["events"] for c in CHANNELS})


if __name__ == "__main__":
    main()
