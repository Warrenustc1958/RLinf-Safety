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

"""Step 1: rebuild per-step, per-channel SAM labels from the audit grades.

Does not read the Wan label NPZ for safety (it is only cross-checked). For every
manifest row of cws_v1 / cws_test_new the root prefix and branch suffix audits are
stitched (root[:bt-1] + branch) and truncated to the row's num_frames - 1 actions.

Rules: a rule is identified by (source, predicate, hazard). The expected rules of a
segment are its violation predicates (checkcontact / checkrobotcontact /
wronggrasppart) plus every grade seen in that segment (aux hand contact has no
predicate). Per step and channel:
  cost = OR of the channel's rule ``violated``
  valid_channel = cost or (channel has expected rules and all are graded this step)
A step without ``grades`` (legacy audits) is invalid on every channel.

Writes <out>/labels/<uid>.npz, <out>/episodes.jsonl, <out>/audit.jsonl (reason
codes), <out>/report.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from toolkits.libero_safety.sam_alignment.schema import (
    CH, CHANNELS, DIST_MAX, INDEX_ROOT, INDEXES, K, OUT_ROOT, PRED_CHANNEL, audit_json_of_npz, group_key, uid,
)


def rule_of_pred(p: dict):
    """(rule key, channel, hazard). Official constraints are keyed by their instance id
    (one hazard can carry several checkcontact rules); aux rules by (predicate, hazard)."""
    pred = str(p["predicate"]).lower()
    if p["source"] not in ("constraint", "aux_constraint") or pred not in PRED_CHANNEL:
        return None
    args = p["arguments"]
    hazard = str(args[0] if pred == "wronggrasppart" else args[-1])
    key = (p["source"], p["instance_id"]) if p["source"] == "constraint" else (p["source"], pred, hazard)
    return key, PRED_CHANNEL[pred], hazard


def rule_of_grade(g: dict):
    head, pred = g["name"].split("|")[:2]
    key = (g["source"], head) if g["source"] == "constraint" else (g["source"], pred.lower(), str(g["hazard"]))
    return key, g["channel"], str(g["hazard"])


def segment_labels(traj: list[dict]) -> dict:
    """Per-step arrays for one audit segment (list of transitions)."""
    T = len(traj)
    cost = np.zeros((T, K), bool)
    valid = np.zeros((T, K), bool)
    dist = np.full((T, K), np.nan, np.float32)
    dist_valid = np.zeros((T, K), bool)
    censored = np.zeros((T, K), bool)
    near = np.zeros((T, K), bool)
    force = np.zeros((T, K), np.float32)
    hazard = np.full((T, K), -1, np.int16)
    official_violation = np.array([bool(s["terminations"]["safety_violation"]) for s in traj])
    success = np.array([bool(s["terminations"]["raw_task"]) for s in traj])
    action = np.array([s["action"] for s in traj], np.float32)
    has_grades = np.array(["grades" in s for s in traj])
    codes = collections.Counter()

    expected = {}  # rule key -> (channel, hazard)
    for s in traj[:1]:
        for p in s["predicates"]:
            r = rule_of_pred(p)
            if r:
                expected[r[0]] = (r[1], r[2])
    for s in traj:
        for g in s.get("grades", []):
            r, ch, hz_name = rule_of_grade(g)
            expected.setdefault(r, (ch, hz_name))
    hazards = sorted({h for _, h in expected.values()})
    hz = {h: i for i, h in enumerate(hazards)}
    per_ch = collections.defaultdict(set)
    for r, (ch, _) in expected.items():
        per_ch[ch].add(r)

    for t, s in enumerate(traj):
        if not has_grades[t]:
            continue
        gm = {}
        for g in s["grades"]:
            gm[rule_of_grade(g)[0]] = g
        pv = {}
        for p in s["predicates"]:
            r = rule_of_pred(p)
            if r:
                pv[r[0]] = bool(p["value"])
        for ch_name, rules in per_ch.items():
            k = CH[ch_name]
            present = [gm[r] for r in rules if r in gm]
            missing = len(rules) - len(present)
            viol = [g for g in present if g["violated"]]
            cost[t, k] = bool(viol)
            valid[t, k] = bool(viol) or (missing == 0 and bool(rules))
            if missing:
                codes[f"missing_rule_steps:{ch_name}"] += 1
            if viol:
                hazard[t, k] = hz[str(viol[0]["hazard"])]
            if present:
                near[t, k] = any(g["near_miss"] for g in present)
                force[t, k] = float(sum(g["normal_force"] for g in present))
                if ch_name != "c_grasp":
                    d = min(float(g["min_dist"]) for g in present)
                    dist[t, k] = d
                    dist_valid[t, k] = missing == 0
                    censored[t, k] = d >= DIST_MAX - 1e-4
            for r in rules:
                if r in pv and r in gm and pv[r] != bool(gm[r]["violated"]):
                    codes[f"official_pred_neq_grade:{ch_name}"] += 1
        if bool(cost[t].any()) != bool(official_violation[t]):
            codes["any_grade_neq_safety_violation"] += 1
    return dict(cost=cost, valid=valid, dist=dist, dist_valid=dist_valid, censored=censored, near=near,
                force=force, hazard=hazard, official_violation=official_violation, success=success,
                action=action, has_grades=has_grades, hazards=hazards, codes=codes)


def concat(a: dict, b: dict, p: int) -> dict:
    """root[:p] + branch; hazard ids remapped onto a merged hazard table."""
    hazards = sorted(set(a["hazards"]) | set(b["hazards"]))
    hz = {h: i for i, h in enumerate(hazards)}
    out = {}
    for key in ("cost", "valid", "dist", "dist_valid", "censored", "near", "force", "official_violation",
                "success", "action", "has_grades"):
        out[key] = np.concatenate([a[key][:p], b[key]], 0)
    remap = []
    for seg, take in ((a, slice(0, p)), (b, slice(None))):
        h = seg["hazard"][take].copy()
        lut = np.array([hz[x] for x in seg["hazards"]] + [-1], np.int16)
        remap.append(np.where(h >= 0, lut[h], -1).astype(np.int16))
    out["hazard"] = np.concatenate(remap, 0)
    out["hazards"] = hazards
    out["codes"] = a["codes"] + b["codes"]
    return out


def first_true(x):
    i = np.flatnonzero(x)
    return int(i[0]) if len(i) else None


def snapshot_state(audit_path: str, j: dict, timestep: int):
    s = j.get("sim_state_trajectory")
    if not isinstance(s, dict):
        return None
    try:
        z = np.load(os.path.join(os.path.dirname(audit_path), s["path"]))
        ts = list(z["timesteps"])
        return z["mujoco_state"][ts.index(timestep)] if timestep in ts else None
    except Exception:
        return None


def process(task):
    index, row, out_dir = task
    u = uid(index, row["idx"])
    try:
        bj = json.load(open(row["json"]))
        n_act = row["num_frames"] - 1
        fingerprint = None
        if row["kind"] == "root":
            lab = segment_labels(bj["trajectory"])
            root_json = row["json"]
        else:
            bt = int(row["branch_timestep"])
            root_json = audit_json_of_npz(row["agent_segments"][0][0])
            rj = json.load(open(root_json))
            lab = concat(segment_labels(rj["trajectory"]), segment_labels(bj["trajectory"]), bt - 1)
            a, b = snapshot_state(root_json, rj, bt), snapshot_state(row["json"], bj, 1)
            if a is not None and b is not None:
                fingerprint = float(np.abs(a - b).max())
        codes = lab.pop("codes")
        hazards = lab.pop("hazards")
        T = len(lab["cost"])
        if T < n_act:
            raise ValueError(f"stitched {T} actions < num_frames-1 = {n_act}")
        full_cost = lab["cost"]
        lab = {k: v[:n_act] for k, v in lab.items()}
        valid_step = np.ones(n_act, bool)
        legacy = not lab["has_grades"].all()
        if legacy:
            codes["legacy_no_grades_steps"] += int((~lab["has_grades"]).sum())
        for k, c in enumerate(CHANNELS):
            if full_cost[n_act:, k].any() and not lab["cost"][:, k].any():
                codes[f"event_only_after_trim:{c}"] += 1
        with np.load(row["labels"]) as w:
            wv = w["violation"][:n_act]
            if not np.array_equal(wv, lab["official_violation"]):
                codes["wan_npz_violation_neq_audit"] += 1
            if not np.allclose(w["actions"][:n_act], lab["action"], atol=1e-6):
                codes["wan_npz_action_neq_audit"] += 1
        success = bool(lab["success"].any())
        rule_unsafe = bool((lab["cost"] & lab["valid"]).any())
        actual_q = None if legacy else {(True, False): "Q1", (True, True): "Q2", (False, False): "Q3",
                                        (False, True): "Q4"}[(success, rule_unsafe)]
        if actual_q is not None and actual_q != row["quadrant"]:
            codes[f"quadrant_manifest_{row['quadrant']}_actual_{actual_q}"] += 1
        if legacy and row["quadrant"] in ("Q2", "Q4") and not lab["official_violation"].any():
            codes["legacy_unsafe_quadrant_but_no_violation_steps"] += 1
        path = out_dir / "labels" / (u.replace("/", "__") + ".npz")
        np.savez_compressed(path, cost=lab["cost"], valid_step=valid_step, valid_channel=lab["valid"],
                            distance=lab["dist"], distance_valid=lab["dist_valid"],
                            distance_censored=lab["censored"], near_miss=lab["near"], normal_force=lab["force"],
                            hazard=lab["hazard"], success=lab["success"], action_env=lab["action"],
                            official_violation=lab["official_violation"])
        first_event = {c: first_true(lab["cost"][:, k] & lab["valid"][:, k]) for k, c in enumerate(CHANNELS)}
        ep = {
            "uid": u, "index": index, "idx": row["idx"], "manifest_split": row["split"], "src": row["src"],
            "group": group_key(row), "tcell": row["tcell"], "suite": row["suite"], "level": row["level"],
            "task": row["task"], "trial": row["trial"], "kind": row["kind"], "json": row["json"],
            "root_json": root_json, "branch_timestep": row["branch_timestep"],
            "scene_mutated": bool(bj["episode"].get("scene_mutated", False)),
            "state_fingerprint_maxdiff": fingerprint, "num_frames": row["num_frames"], "num_actions": n_act,
            "dual": bool(row["dual"]), "agent_segments": row["agent_segments"],
            "wrist_segments": row["wrist_segments"], "quadrant_manifest": row["quadrant"],
            "actual_quadrant": actual_q, "success": success, "label_source": "legacy" if legacy else "grader_v1",
            "hazards": hazards, "first_event_by_channel": first_event, "labels": str(path),
        }
        return ep, {"uid": u, "codes": dict(codes)}
    except Exception:
        return None, {"uid": u, "error": traceback.format_exc(limit=4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT_ROOT / "labels"))
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="first N rows per index (pilot)")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    tasks = []
    for index in INDEXES:
        rows = [json.loads(l) for l in (INDEX_ROOT / index / "manifest.jsonl").read_text().splitlines()]
        if args.limit:
            rows = rows[: args.limit]
        tasks += [(index, r, out) for r in rows]
    t0 = time.time()
    eps, audits = [], []
    with Pool(args.workers) as pool:
        for i, (ep, au) in enumerate(pool.imap_unordered(process, tasks, chunksize=4)):
            audits.append(au)
            if ep:
                eps.append(ep)
            if (i + 1) % 1000 == 0:
                print(f"{i + 1}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    eps.sort(key=lambda e: e["uid"])
    by_json = {e["json"]: e["uid"] for e in eps if e["kind"] == "root"}
    for e in eps:
        e["root_uid"] = e["uid"] if e["kind"] == "root" else by_json.get(e["root_json"])
    with open(out / "episodes.jsonl", "w") as f:
        for e in eps:
            f.write(json.dumps(e) + "\n")
    with open(out / "audit.jsonl", "w") as f:
        for a in sorted(audits, key=lambda a: a["uid"]):
            f.write(json.dumps(a) + "\n")

    # report: label counts per source / channel, events, reason codes
    rep = {"rows": len(tasks), "episodes": len(eps), "errors": sum("error" in a for a in audits),
           "seconds": round(time.time() - t0)}
    rep["label_source"] = dict(collections.Counter((e["index"], e["src"], e["label_source"]) and
                                                   f"{e['index']}|{e['src']}|{e['label_source']}" for e in eps))
    codes = collections.Counter()
    for a in audits:
        for c, v in a.get("codes", {}).items():
            codes[c] += v
        if a.get("codes"):
            for c in a["codes"]:
                codes["episodes_with:" + c] += 1
    rep["reason_codes"] = dict(sorted(codes.items()))
    stats = collections.defaultdict(lambda: np.zeros(5, np.int64))  # pos, neg, missing, events, eps_with_event
    for e in eps:
        if e["label_source"] != "grader_v1":
            continue
        with np.load(e["labels"]) as z:
            c, v = z["cost"], z["valid_channel"]
        for k, ch in enumerate(CHANNELS):
            pos = c[:, k] & v[:, k]
            ev = int((pos[1:] & ~pos[:-1]).sum() + pos[0]) if len(pos) else 0
            for key in (f"{e['index']}|{ch}", f"{e['index']}|{e['suite']}|{ch}"):
                stats[key] += [int(pos.sum()), int((v[:, k] & ~c[:, k]).sum()), int((~v[:, k]).sum()), ev, int(ev > 0)]
    rep["channel_counts"] = {k: dict(zip(("pos_steps", "neg_steps", "invalid_steps", "events", "episodes_with_event"),
                                         map(int, v))) for k, v in sorted(stats.items())}
    rep["quadrant_actual_vs_manifest"] = dict(collections.Counter(
        f"{e['index']}|{e['quadrant_manifest']}->{e['actual_quadrant']}" for e in eps))
    fp = [e["state_fingerprint_maxdiff"] for e in eps if e["kind"] != "root"]
    rep["branch_fingerprint"] = {"branches": len(fp), "missing": sum(x is None for x in fp),
                                 "max": max((x for x in fp if x is not None), default=None),
                                 "nonzero": sum(1 for x in fp if x is not None and x > 1e-6)}
    rep["branch_root_found"] = sum(1 for e in eps if e["kind"] != "root" and e["root_uid"])
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("rows", "episodes", "errors", "seconds", "label_source")}, indent=1))


if __name__ == "__main__":
    main()
