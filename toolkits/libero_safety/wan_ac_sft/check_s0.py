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

"""S0 gate checks over a built index; writes <index>/s0_report.json.

G1 stitching: for sampled roots and depth-1 branches, stitched frames/actions must
   equal ``build_dataset._stitch_sequence``; for every sampled branch (incl. nested)
   the seam drift |parent_frame[bt-1] - branch_frame[0]| must be <= 2.0.
G2 split: no (suite, level, task, trial) in two splits; per task val/test counts.
G3 sampler: 1e5 draws -> quadrant shares, mode shares, start coverage.
G4 dual subset count and agreement with the wrist re-render ok list.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from toolkits.libero_safety import build_dataset as bd
from toolkits.libero_safety.wan_ac_sft.sampler import QUADRANTS, WindowSampler, load_frames

DRIFT_TOL = bd.BRANCH_FRAME_MATCH_TOLERANCE


def g1_stitch(rows, by_key, n_sample, seed):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in rows:
        kind = "standalone" if not r["prefix_available"] else min(r["chain_depth"], 2)
        groups[(str(kind), r["quadrant"])].append(r)
    sample = []
    for g in sorted(groups):
        sample += rng.sample(groups[g], min(len(groups[g]), max(1, n_sample // len(groups))))
    nested = [r for r in rows if r["chain_depth"] >= 2]
    sample += rng.sample(nested, min(len(nested), 100))
    res = {"checked": 0, "exact_vs_build_dataset": 0, "exact_fail": [], "seam_checked": 0,
           "seam_max_drift": 0.0, "seam_fail": [], "len_fail": []}
    for r in sample:
        res["checked"] += 1
        lab = np.load(r["labels"])
        n = r["num_frames"]
        if len(lab["actions"]) != n - 1 or len(lab["eef"]) != n:
            res["len_fail"].append(r["json"])
        if r["chain_depth"] >= 1:
            parent = by_key[(str(Path(r["json"]).parent), r["parent"])]
            bt = int(r["branch_timestep"])
            pf = load_frames(parent["agent_segments"], [bt - 1], "agentview_image")[0]
            bf = load_frames(r["agent_segments"], [bt - 1], "agentview_image")[0]
            drift = float(np.abs(pf.astype(np.int16) - bf.astype(np.int16)).mean())
            res["seam_checked"] += 1
            res["seam_max_drift"] = max(res["seam_max_drift"], drift)
            if drift > DRIFT_TOL:
                res["seam_fail"].append([r["json"], drift])
        if r["chain_depth"] <= 1:
            ep = bd.validate_episode(Path(r["json"]), "agentview_image", "c")
            root = None
            if r["chain_depth"] == 1:
                root = bd.validate_episode(Path(by_key[(str(Path(r["json"]).parent), r["parent"])]["json"]),
                                           "agentview_image", "c")
            ref = bd._stitch_sequence(ep, root)
            mine = load_frames(r["agent_segments"], list(range(n)), "agentview_image")
            ref_act = np.asarray([s["action"] for s in ref.steps], np.float32)
            if mine.shape == ref.images.shape and np.array_equal(mine, ref.images) \
                    and np.array_equal(lab["actions"], ref_act):
                res["exact_vs_build_dataset"] += 1
            else:
                res["exact_fail"].append(r["json"])
    res["exact_total"] = sum(1 for r in sample if r["chain_depth"] <= 1)
    res["pass"] = not (res["exact_fail"] or res["seam_fail"] or res["len_fail"])
    return res


def g2_split(rows, n_val, n_test):
    side = defaultdict(set)
    for r in rows:
        side[(r["suite"], r["level"], r["task"], r["trial"])].add(r["split"])
    cross = [list(k) for k, v in side.items() if len(v) > 1]
    per_task = defaultdict(lambda: {"train": set(), "val": set(), "test": set()})
    for (s, l, t, tr), v in side.items():
        per_task[(s, l, t)][next(iter(v))].add(tr)
    bad_counts = {"|".join(map(str, k)): {p: len(x) for p, x in v.items()} for k, v in per_task.items()
                  if len(v["val"]) != n_val or len(v["test"]) != n_test or not v["train"]}
    eps = Counter(r["split"] for r in rows)
    q = {p: Counter(r["quadrant"] for r in rows if r["split"] == p) for p in ("train", "val", "test")}
    return {"cross_split_init_states": cross, "bad_task_counts": bad_counts, "tasks": len(per_task),
            "episodes": dict(eps), "quadrants": {p: dict(c) for p, c in q.items()},
            "pass": not cross and not bad_counts}


def g3_sampler(rows, draws, seed):
    s = WindowSampler([r for r in rows if r["split"] == "train"])
    rng = np.random.default_rng(seed)
    q, mode, late, long_cov = Counter(), Counter(), 0, []
    for _ in range(draws):
        row, ids, m = s.draw(rng)
        q[row["quadrant"]] += 1
        mode[m] += 1
        if ids[1] > 262:
            late += 1
        if row["num_frames"] >= 500 and m == "uniform":
            long_cov.append((ids[1], row["num_frames"] - 12))
    target = 1.0 / len(s.quadrants)
    shares = {k: q[k] / draws for k in QUADRANTS}
    q_ok = all(abs(shares[k] - target) <= 0.02 for k in s.quadrants)
    with_events = sum(1 for r in s.rows if s.events(r))
    max_reach = max((st / hi for st, hi in long_cov), default=0.0)
    return {"draws": draws, "quadrant_share": shares, "mode_share": {k: v / draws for k, v in mode.items()},
            "starts_beyond_262": late / draws, "long_episode_max_start_over_limit": max_reach,
            "rows_with_events": with_events, "rows": len(s.rows), "dropped_short": s.dropped_short,
            "pass": q_ok and max_reach > 0.98}


def g4_dual(rows, wrist_results, expected):
    ok = set()
    for line in Path(wrist_results).read_text().splitlines():
        r = json.loads(line)
        (ok.add if r["status"] == "ok" else ok.discard)(str(Path(r["json"]).resolve()))
    n_dual = sum(r["dual"] for r in rows)
    own_ok_not_dual = sum(1 for r in rows if r["json"] in ok and not r["dual"])
    return {"dual": n_dual, "expected": expected, "own_wrist_ok": sum(r["json"] in ok for r in rows),
            "own_ok_but_ancestor_missing": own_ok_not_dual, "pass": n_dual == expected}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("--wrist-results", required=True)
    ap.add_argument("--expected-dual", type=int, default=4585)
    ap.add_argument("--n-sample", type=int, default=100)
    ap.add_argument("--draws", type=int, default=100_000)
    ap.add_argument("--n-val", type=int, default=2)
    ap.add_argument("--n-test", type=int, default=3)
    args = ap.parse_args()
    rows = [json.loads(l) for l in (Path(args.index) / "manifest.jsonl").read_text().splitlines()]
    by_key = {(str(Path(r["json"]).parent), r["episode_id"]): r for r in rows}
    rep = {"G1_stitch": g1_stitch(rows, by_key, args.n_sample, 0),
           "G2_split": g2_split(rows, args.n_val, args.n_test),
           "G3_sampler": g3_sampler(rows, args.draws, 0),
           "G4_dual": g4_dual(rows, args.wrist_results, args.expected_dual)}
    standalone = [r for r in rows if r["kind"] != "root" and not r["prefix_available"]]
    rep["standalone_branches"] = {"n": len(standalone),
                                  "quadrants": dict(Counter(r["quadrant"] for r in standalone)),
                                  "dual": sum(r["dual"] for r in standalone)}
    rep["all_pass"] = all(v["pass"] for v in rep.values() if isinstance(v, dict) and "pass" in v)
    (Path(args.index) / "s0_report.json").write_text(json.dumps(rep, indent=1, default=str))
    for k, v in rep.items():
        if isinstance(v, dict) and "pass" in v:
            print(k, "PASS" if v["pass"] else "FAIL",
                  json.dumps({a: b for a, b in v.items() if a not in ("exact_fail", "seam_fail", "len_fail",
                              "cross_split_init_states", "bad_task_counts")}, default=str)[:600])
    print("ALL_PASS", rep["all_pass"])


if __name__ == "__main__":
    main()
