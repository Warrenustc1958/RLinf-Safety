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

"""Step 2: split registry by initial-state group (suite, level, task_id, trial_id).

Development roles come from the grader-labelled cws_v1 train groups only, split per
task cell into cost_train / h_train / dev (default 70 / 15 / 15). All roots,
branches and pairs of a group stay in one role. Held-out sets already used for
Wan / RM selection are registered as diagnostics, not as fresh test sets.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from toolkits.libero_safety.sam_alignment.schema import CHANNELS, OUT_ROOT


def stable(seed: int, key: str) -> int:
    return int(hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()[:12], 16)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", default=str(OUT_ROOT / "labels"))
    ap.add_argument("--out", default=str(OUT_ROOT / "splits" / "split_registry.json"))
    ap.add_argument("--fractions", nargs=3, type=float, default=[0.70, 0.15, 0.15])
    ap.add_argument("--seed", type=int, default=20261024)
    args = ap.parse_args()

    eps = [json.loads(l) for l in (Path(args.labels) / "episodes.jsonl").read_text().splitlines()]
    role_of_ep = {}
    dev_pool = collections.defaultdict(set)  # tcell -> groups
    for e in eps:
        if e["index"] == "cws_v1" and e["manifest_split"] == "train":
            if e["label_source"] == "grader_v1" and e["dual"]:
                dev_pool[e["tcell"]].add(e["group"])
            else:
                role_of_ep[e["uid"]] = "excluded_legacy_train"
        elif e["index"] == "cws_v1":
            role_of_ep[e["uid"]] = f"diag_legacy_{e['manifest_split']}"
        else:
            role_of_ep[e["uid"]] = f"diag_new{e['manifest_split']}"

    names = ("cost_train", "h_train", "dev")
    cum = np.cumsum(args.fractions) / sum(args.fractions)
    group_role = {}
    for tcell in sorted(dev_pool):
        groups = sorted(dev_pool[tcell], key=lambda g: stable(args.seed, g))
        n = len(groups)
        for i, g in enumerate(groups):
            group_role[g] = names[int(np.searchsorted(cum, (i + 0.5) / n))]
    for e in eps:
        if e["uid"] not in role_of_ep:
            role_of_ep[e["uid"]] = group_role[e["group"]]

    # a group may also appear in legacy rows (old_fill) of the same index; keep them excluded
    roles = collections.defaultdict(lambda: {"groups": set(), "episodes": 0, "roots": 0, "branches": 0})
    for e in eps:
        r = roles[role_of_ep[e["uid"]]]
        r["groups"].add(e["group"])
        r["episodes"] += 1
        r["roots" if e["kind"] == "root" else "branches"] += 1

    # leakage checks between roles that are compared with each other
    checks = {}
    core = [r for r in ("cost_train", "h_train", "dev") if r in roles]
    for i, a in enumerate(core):
        for b in core[i + 1:]:
            checks[f"{a}&{b}"] = len(roles[a]["groups"] & roles[b]["groups"])
    for d in [r for r in roles if r.startswith("diag")]:
        for a in core:
            checks[f"{a}&{d}"] = len(roles[a]["groups"] & roles[d]["groups"])
    pairs_cross = 0
    by_uid = {e["uid"]: e for e in eps}
    for e in eps:
        if e["kind"] != "root" and e.get("root_uid") and role_of_ep[e["root_uid"]] != role_of_ep[e["uid"]]:
            if role_of_ep[e["uid"]] in core:
                pairs_cross += 1
    checks["branch_root_role_mismatch_in_core"] = pairs_cross

    counts = collections.defaultdict(lambda: np.zeros((len(CHANNELS), 3), np.int64))
    for e in eps:
        if e["label_source"] != "grader_v1":
            continue
        with np.load(e["labels"]) as z:
            c, v = z["cost"], z["valid_channel"]
        pos = c & v
        ev = (pos[1:] & ~pos[:-1]).sum(0) + pos[0] if len(pos) else np.zeros(len(CHANNELS))
        counts[role_of_ep[e["uid"]]] += np.stack([pos.sum(0), ev, (ev > 0)], 1)
    per_role_groups_with_event = collections.defaultdict(lambda: collections.defaultdict(set))
    for e in eps:
        if e["label_source"] == "grader_v1":
            for c, t in e["first_event_by_channel"].items():
                if t is not None:
                    per_role_groups_with_event[role_of_ep[e["uid"]]][c].add(e["group"])

    registry = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": args.seed, "fractions": args.fractions,
        "group_key": "(suite, level, task_id, trial_id)",
        "labels_dir": args.labels,
        "roles": {
            name: {"groups": len(r["groups"]), "episodes": r["episodes"], "roots": r["roots"],
                   "branches": r["branches"],
                   "channels": {c: {"pos_steps": int(counts[name][k, 0]), "events": int(counts[name][k, 1]),
                                    "episodes_with_event": int(counts[name][k, 2]),
                                    "groups_with_event": len(per_role_groups_with_event[name][c])}
                                for k, c in enumerate(CHANNELS)} if name in counts else None}
            for name, r in sorted(roles.items())},
        "usage_history": {
            "cost_train": "SAM base cost training (Step 4/6)",
            "h_train": "reserved: never used for base cost fitting or selection (Step 7)",
            "dev": "SAM development / model selection",
            "diag_newval": "used to select RM rm_v1 and in Wan eval_v2_dual_newtest; diagnostic only",
            "diag_newtest": "RM rm_v1 test and Wan eval_v2_dual_newtest; diagnostic only",
            "diag_legacy_val": "legacy audits without grades; RM selection and Wan eval; no channel labels",
            "diag_legacy_test": "legacy audits without grades; RM test and Wan eval; no channel labels",
            "excluded_legacy_train": "old_fill / non-dual train rows without trustworthy channel labels",
            "calib_fresh": "PENDING: needs new initial-state groups collected after the design is frozen",
            "test_fresh": "PENDING: needs new initial-state groups collected after the design is frozen",
        },
        "leakage_checks": checks,
        "group_role": dict(sorted(group_role.items())),
        "episode_role": dict(sorted(role_of_ep.items())),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(registry, indent=1))
    print(json.dumps({"roles": {k: {kk: v[kk] for kk in ("groups", "episodes")} for k, v in registry["roles"].items()},
                      "leakage_checks": checks}, indent=1))


if __name__ == "__main__":
    main()
