"""CWS index, phase 1: stitch, label and profile every admitted step_e episode.

Source: step_e/archive/outcomes.jsonl rows with admitted=true. Episodes whose
(task, trial_id) is an old v1 test/val init state are skipped (no leakage into the
held-out sets). Branches are stitched as root[:bt-1] + branch (the on-disk branch is
the post-branch suffix and its frame 0 equals root frame bt-1). A branch whose root
violated before the branch point is upgraded to unsafe.

Per episode writes
  <out>/labels/<idx>.npz   actions [n-1, 7], violation [n-1], success [n-1]
  <out>/motion/<idx>.npy   motion[s] = mean|f[s+11] - f[s+4]| (0-255), s = 1..n-12
  <out>/stats.jsonl        manifest fields + dup / span / tail-static stats
Phase 2 (select_cws.py) applies the static filter and the old-data fill.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

NEW = "/data_cf/cws/step_e/archive/outcomes.jsonl"
V1 = "/mnt/nfs/fjj/wan_ac_sft/index/v1"
QUAD = {(True, False): "Q1", (True, True): "Q2", (False, False): "Q3", (False, True): "Q4"}
OUT = None


def npz_of(audit_path):
    d, n = os.path.split(audit_path)
    return os.path.join(d, "visual", n.replace(".json", ".npz"))


def load_ep(path):
    j = json.load(open(path))
    tr = j["trajectory"]
    act = np.array([s["action"] for s in tr], np.float32)
    viol = np.array([bool(s["terminations"]["safety_violation"]) for s in tr])
    succ = np.array([bool(s["terminations"]["raw_task"]) for s in tr])
    return j, act, viol, succ


def profile(frames):
    """Consecutive diff, 8-frame span profile, computed in blocks to bound memory."""
    n = len(frames)
    d1 = np.empty(n - 1, np.float32)
    for a in range(0, n - 1, 64):
        b = min(a + 64, n - 1)
        d1[a:b] = np.abs(frames[a + 1:b + 1].astype(np.int16) - frames[a:b].astype(np.int16)).mean(axis=(1, 2, 3))
    hi = n - 12
    prof = np.full(n + 1, np.nan, np.float32)
    if hi >= 1:
        for a in range(1, hi + 1, 64):
            b = min(a + 64, hi + 1)
            prof[a:b] = np.abs(frames[a + 11:b + 11].astype(np.int16) - frames[a + 4:b + 4].astype(np.int16)).mean(axis=(1, 2, 3))
    return d1, prof


def tail_static(prof, thr=1.0):
    v = prof[np.isfinite(prof)]
    k = 0
    for x in v[::-1]:
        if x < thr:
            k += 1
        else:
            break
    return k


def first_true(a):
    i = np.flatnonzero(a)
    return int(i[0]) + 1 if len(i) else None   # action i produces frame i + 1


def process_group(task):
    root_path, members, held = task
    out = []
    try:
        rj, r_act, r_viol, r_succ = load_ep(root_path)
        meta = rj["episode"]
        tcell = "%s_%s_t%s" % (meta["task_suite"], meta["safety_level"], meta["task_id"])
        trial = int(meta["trial_id"])
        if (tcell, trial) in held:
            return [dict(skip="leak", n=len(members))]
        rz = np.load(npz_of(root_path))
        r_agent = rz["agentview_image"]
        has_wrist = "robot0_eye_in_hand_image" in rz.files
    except Exception:
        return [dict(error=traceback.format_exc(limit=3), path=root_path)]
    for o in members:
        try:
            if o["audit_path"] == root_path:
                j, act, viol, succ, frames, bt = rj, r_act, r_viol, r_succ, r_agent, None
                segs = [[npz_of(root_path), 0, len(r_agent)]]
                inherited_unsafe = False
            else:
                j, b_act, b_viol, b_succ = load_ep(o["audit_path"])
                bt = int(j["episode"]["branch_timestep"])
                p = bt - 1
                b_agent = np.load(npz_of(o["audit_path"]))["agentview_image"]
                if len(r_agent) < p or len(r_act) < p:
                    raise ValueError("root shorter than branch point")
                frames = np.concatenate([r_agent[:p], b_agent], 0)
                act = np.concatenate([r_act[:p], b_act], 0)
                viol = np.concatenate([r_viol[:p], b_viol], 0)
                succ = np.concatenate([r_succ[:p], b_succ], 0)
                segs = [[npz_of(root_path), 0, p], [npz_of(o["audit_path"]), 0, len(b_agent)]]
                inherited_unsafe = bool(r_viol[:p].any())
            n = len(frames)
            assert len(act) == n - 1, (len(act), n)
            success = bool(j["outcome"]["success"])
            unsafe = bool(j["outcome"]["unsafe"]) or inherited_unsafe
            q = QUAD[(success, unsafe)]
            d1, prof = profile(frames)
            idx = o["idx"]
            np.savez(os.path.join(OUT, "labels", idx + ".npz"), actions=act, violation=viol, success=succ)
            with open(os.path.join(OUT, "motion", idx + ".npy"), "wb") as f:
                np.save(f, prof)
            pv = prof[np.isfinite(prof)]
            out.append(dict(
                idx=idx, src="new", json=o["audit_path"], cell="%s_%s" % (meta["task_suite"], meta["safety_level"]),
                tcell=tcell, suite=meta["task_suite"], level=meta["safety_level"], task=int(meta["task_id"]),
                trial=trial, kind="root" if bt is None else "counterfactual_branch", quadrant=q,
                quadrant_collector=o["quadrant"], upgraded=inherited_unsafe and o["quadrant"] in ("Q1", "Q3"),
                branch_timestep=bt, branch_frame=bt, num_frames=n,
                first_violation_frame=first_true(viol), first_success_frame=first_true(succ),
                agent_segments=segs, wrist_segments=segs if has_wrist else None, dual=has_wrist,
                labels=os.path.join(OUT, "labels", idx + ".npz"), prefix_available=True,
                near_miss=bool(o.get("near_miss")), program_id=o.get("program_id"),
                dup=float(np.mean(d1 < 0.5)), span_med=float(np.median(pv)) if len(pv) else None,
                lt2=float(np.mean(pv < 2)) if len(pv) else None, tail_static=tail_static(prof)))
        except Exception:
            out.append(dict(error=traceback.format_exc(limit=3), path=o["audit_path"]))
    return out


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="/data_cf/wan_ac_sft/index/cws_v1")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit_groups", type=int, default=0)
    a = ap.parse_args()
    OUT = a.out
    for d in ("labels", "motion"):
        Path(OUT, d).mkdir(parents=True, exist_ok=True)

    v1 = [json.loads(l) for l in open(os.path.join(V1, "manifest.jsonl"))]
    held = {("%s_%s_t%s" % (r["suite"], r["level"], r["task"]), r["trial"]) for r in v1 if r["split"] in ("test", "val")}

    rows = [json.loads(l) for l in open(NEW)]
    adm = [o for o in rows if o.get("admitted")]
    adm.sort(key=lambda o: o["audit_path"])
    for k, o in enumerate(adm):
        o["idx"] = "n%05d" % k
    groups = collections.OrderedDict()
    for o in adm:
        if o["kind"] == "root":
            rp = o["audit_path"]
        else:
            with open(o["audit_path"], "rb") as f:
                m = re.search(rb'"root_episode_id":\s*"([^"]+)"', f.read(65536))
            rp = os.path.join(os.path.dirname(o["audit_path"]), m.group(1).decode() + ".json")
        groups.setdefault(rp, []).append(o)
    tasks = [(rp, ms, held) for rp, ms in groups.items()]
    if a.limit_groups:
        tasks = tasks[: a.limit_groups]
    print("[cws] admitted %d, groups %d (to process %d), workers %d" % (len(adm), len(groups), len(tasks), a.workers), flush=True)

    t0, n_ok, n_err, n_leak = time.time(), 0, 0, 0
    with open(os.path.join(OUT, "stats.jsonl"), "w") as f, open(os.path.join(OUT, "errors.jsonl"), "w") as fe, Pool(a.workers) as p:
        for k, res in enumerate(p.imap_unordered(process_group, tasks, chunksize=1)):
            for r in res:
                if "error" in r:
                    n_err += 1
                    fe.write(json.dumps(r) + "\n")
                elif "skip" in r:
                    n_leak += r["n"]
                else:
                    n_ok += 1
                    f.write(json.dumps(r) + "\n")
            if k % 200 == 0:
                print("[cws] group %d/%d ok %d leak %d err %d %.0fs" % (k, len(tasks), n_ok, n_leak, n_err, time.time() - t0), flush=True)
    print("[cws] DONE ok %d leak-skipped %d errors %d in %.0fs" % (n_ok, n_leak, n_err, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
