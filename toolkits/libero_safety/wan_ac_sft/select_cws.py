"""CWS index, phase 2: static filter, tail trim, old-data fill, manifest.

Inputs: phase-1 stats (<out>/stats.jsonl), old v1 manifest + motion, kin regrade.
  * static episode: > STATIC_FRAC of its windows move < 2/255 -> dropped, but each
    (task, quadrant) keeps at least MIN_KEEP episodes (least static first);
  * tail trim: trailing windows moving < 1/255 are cut down to TAIL_KEEP, never
    before first violation / success + 16;
  * fill: for each (task, quadrant) with fewer than TARGET new episodes, add old
    train episodes of the same task and regraded quadrant that are <= 513 frames,
    have their prefix, and are not static, highest motion first;
  * old val / test rows are copied unchanged (original quadrant kept so earlier
    evaluations stay comparable; the regraded label is stored as quadrant_regraded).
Writes <out>/manifest.jsonl, <out>/selection_report.json and links old motion files.
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np

V1 = "/mnt/nfs/fjj/wan_ac_sft/index/v1"
KIN = "/data_cf/cws/kin_legacy_5k/rows.jsonl"
Q = ("Q1", "Q2", "Q3", "Q4")
QUAD = {(True, False): "Q1", (True, True): "Q2", (False, False): "Q3", (False, True): "Q4"}


def stat(prof):
    v = prof[np.isfinite(prof)]
    if not len(v):
        return None
    k = 0
    for x in v[::-1]:
        if x < 1.0:
            k += 1
        else:
            break
    return dict(lt2=float(np.mean(v < 2)), span_med=float(np.median(v)), tail_static=k)


def trimmed(r, tail_keep):
    n = r["num_frames"]
    cut = max(0, r["tail_static"] - tail_keep)
    floor = max([13] + [f + 16 for f in (r.get("first_violation_frame"), r.get("first_success_frame")) if f])
    return max(min(n, floor), n - cut)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="/data_cf/wan_ac_sft/index/cws_v1")
    ap.add_argument("--static_frac", type=float, default=0.8)
    ap.add_argument("--min_keep", type=int, default=10)
    ap.add_argument("--target", type=int, default=40)
    ap.add_argument("--tail_keep", type=int, default=8)
    ap.add_argument("--max_frames", type=int, default=513)
    a = ap.parse_args()

    new = [json.loads(l) for l in open(os.path.join(a.out, "stats.jsonl"))]
    v1 = [json.loads(l) for l in open(os.path.join(V1, "manifest.jsonl"))]
    kin = {}
    for l in open(KIN):
        k = json.loads(l)
        kin[k["path"]] = k
    by_epid = {r["episode_id"]: r for r in v1}

    # ---- old: regraded quadrant (+ prefix inheritance), motion stats
    for r in v1:
        r["tcell"] = "%s_%s_t%s" % (r["suite"], r["level"], r["task"])
        k = kin.get(r["json"])
        unsafe = k is not None and k["new_q"] in ("Q2", "Q4")
        if k is not None and r["kind"] != "root" and not unsafe:
            root = by_epid.get(r["parent"])
            rk = kin.get(root["json"]) if root else None
            bt = r.get("branch_timestep")
            unsafe = bool(rk and rk.get("first_viol") is not None and bt and rk["first_viol"] < bt)
        if k is None:
            r["quadrant_regraded"] = r["quadrant"]
        else:
            r["quadrant_regraded"] = QUAD[(r["quadrant"] in ("Q1", "Q2"), unsafe or r["quadrant"] in ("Q2", "Q4"))]
        if (r["first_violation_frame"] is None and r["quadrant_regraded"] in ("Q2", "Q4")
                and r["kind"] == "root" and k and k.get("first_viol")):
            r["first_violation_frame_regraded"] = int(k["first_viol"])
        p = os.path.join(V1, "motion", "%s.npy" % r["idx"])
        s = stat(np.load(p)) if os.path.exists(p) else None
        r.update(s or dict(lt2=None, span_med=None, tail_static=0))

    # ---- new: static filter with per-(task, quadrant) floor
    cell = collections.defaultdict(list)
    for r in new:
        cell[(r["tcell"], r["quadrant"])].append(r)
    keep_new, drop_new = [], []
    for key, rs in cell.items():
        rs.sort(key=lambda r: (r["lt2"] if r["lt2"] is not None else 1.0))
        ok = [r for r in rs if r["lt2"] is not None and r["lt2"] <= a.static_frac]
        bad = [r for r in rs if r not in ok]
        if len(ok) < a.min_keep:
            take = bad[: a.min_keep - len(ok)]
            ok, bad = ok + take, bad[len(take):]
        keep_new += ok
        drop_new += bad

    # ---- old fill
    have = collections.Counter((r["tcell"], r["quadrant"]) for r in keep_new)
    pool = collections.defaultdict(list)
    for r in v1:
        if (r["split"] == "train" and r["num_frames"] <= a.max_frames and r.get("prefix_available") is not False
                and r["lt2"] is not None and r["lt2"] <= a.static_frac):
            pool[(r["tcell"], r["quadrant_regraded"])].append(r)
    tasks = sorted(set(r["tcell"] for r in v1))
    fill, fill_log = [], {}
    for t in tasks:
        for q in Q:
            need = a.target - have.get((t, q), 0)
            cand = sorted(pool.get((t, q), []), key=lambda r: -r["span_med"])
            take = cand[: max(0, need)]
            fill += take
            if need > 0:
                fill_log["%s %s" % (t, q)] = dict(new=have.get((t, q), 0), need=need, old_added=len(take),
                                                   old_available=len(cand))

    # ---- write manifest
    link = os.path.join(a.out, "motion")
    out_rows = []
    for r in keep_new:
        r = dict(r)
        r["num_frames_raw"], r["num_frames"] = r["num_frames"], trimmed(r, a.tail_keep)
        r["split"] = "train"
        out_rows.append(r)
    for r in fill:
        r = dict(r)
        old_idx = r["idx"]
        r["idx"], r["src"] = "o%s" % old_idx, "old_fill"
        r["quadrant_original"], r["quadrant"] = r["quadrant"], r["quadrant_regraded"]
        if r.get("first_violation_frame_regraded"):
            r["first_violation_frame"] = r["first_violation_frame_regraded"]
        r["num_frames_raw"], r["num_frames"] = r["num_frames"], trimmed(r, a.tail_keep)
        r["split"] = "train"
        dst = os.path.join(link, "%s.npy" % r["idx"])
        if not os.path.exists(dst):
            os.symlink(os.path.join(V1, "motion", "%s.npy" % old_idx), dst)
        out_rows.append(r)
    for r in v1:
        if r["split"] in ("val", "test"):
            r = dict(r)
            old_idx = r["idx"]
            r["idx"], r["src"] = "o%s" % old_idx, "old_heldout"
            dst = os.path.join(link, "%s.npy" % r["idx"])
            if not os.path.exists(dst) and os.path.exists(os.path.join(V1, "motion", "%s.npy" % old_idx)):
                os.symlink(os.path.join(V1, "motion", "%s.npy" % old_idx), dst)
            out_rows.append(r)
    with open(os.path.join(a.out, "manifest.jsonl"), "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, default=float) + "\n")

    # ---- report
    tr = [r for r in out_rows if r["split"] == "train"]
    qc = lambda rs: {q: sum(1 for r in rs if r["quadrant"] == q) for q in Q}
    rep = dict(
        params=vars(a),
        new_in=len(new), new_dropped_static=len(drop_new), new_kept=len(keep_new),
        old_fill=len(fill), train=len(tr), val=sum(r["split"] == "val" for r in out_rows),
        test=sum(r["split"] == "test" for r in out_rows),
        train_quadrants=qc(tr), new_kept_quadrants=qc(keep_new),
        fill_quadrants=qc([dict(quadrant=r["quadrant_regraded"]) for r in fill]),
        train_frames=int(sum(r["num_frames"] for r in tr)),
        train_frames_untrimmed=int(sum(r.get("num_frames_raw", r["num_frames"]) for r in tr)),
        train_windows=int(sum(max(0, r["num_frames"] - 12) for r in tr)),
        dual=sum(bool(r.get("dual")) for r in tr),
        per_cell={c: qc([r for r in tr if r["cell"] == c]) for c in sorted(set(r["cell"] for r in tr))},
        per_task={t: dict(new=qc([r for r in keep_new if r["tcell"] == t]),
                          old=qc([dict(quadrant=r["quadrant_regraded"]) for r in fill if r["tcell"] == t]))
                  for t in tasks},
        still_short={k: v for k, v in fill_log.items() if v["new"] + v["old_added"] < a.min_keep},
        fill_log=fill_log)
    json.dump(rep, open(os.path.join(a.out, "selection_report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("per_task", "fill_log")}, indent=1))


if __name__ == "__main__":
    main()
