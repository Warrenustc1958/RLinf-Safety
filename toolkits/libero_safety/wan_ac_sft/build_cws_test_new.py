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

"""New-recording held-out index: step_e episodes on the old v1 test/val init states.

build_cws_index.py skips every admitted step_e episode whose (task, trial_id) is an
old v1 test/val init state, so none of them is in cws_v1 train. Their wrist view is
recorded live (the old held-out wrist is a re-render), and they share init states
with the old held-out set, so old (re-rendered) vs new (recorded) wrist can be
compared on the same starting states.

Writes <out>/{labels,motion,stats.jsonl,errors.jsonl,manifest.jsonl,report.json};
manifest rows get split = test / val after the old v1 split of their init state.
Episodes are not trimmed (same as the old held-out rows).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import time
from multiprocessing import Pool
from pathlib import Path

from toolkits.libero_safety.wan_ac_sft import build_cws_index as bci


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="/data_cf/wan_ac_sft/index/cws_test_new")
    ap.add_argument("--train_index", default="/data_cf/wan_ac_sft/index/cws_v1")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out)
    for d in ("labels", "motion"):
        (out / d).mkdir(parents=True, exist_ok=True)
    bci.OUT = str(out)  # read by process_group in the forked workers

    v1 = [json.loads(l) for l in open(os.path.join(bci.V1, "manifest.jsonl"))]
    split_of = {("%s_%s_t%s" % (r["suite"], r["level"], r["task"]), r["trial"]): r["split"]
                for r in v1 if r["split"] in ("test", "val")}
    train = [json.loads(l) for l in open(os.path.join(a.train_index, "manifest.jsonl"))]
    train_json = {r["json"] for r in train if r["split"] == "train" and r.get("json")}
    train_tt = {(r["tcell"], r["trial"]) for r in train if r["split"] == "train"}

    adm = [json.loads(l) for l in open(bci.NEW)]
    adm = [o for o in adm if o.get("admitted")]
    adm.sort(key=lambda o: o["audit_path"])
    for k, o in enumerate(adm):
        o["idx"] = "h%05d" % k
    groups = collections.OrderedDict()
    for o in adm:
        if o["kind"] == "root":
            rp = o["audit_path"]
        else:
            with open(o["audit_path"], "rb") as f:
                m = re.search(rb'"root_episode_id":\s*"([^"]+)"', f.read(65536))
            rp = os.path.join(os.path.dirname(o["audit_path"]), m.group(1).decode() + ".json")
        groups.setdefault(rp, []).append(o)

    def init_state(rp):
        with open(rp, "rb") as f:
            h = f.read(65536)
        g = lambda k: re.search(rb'"%s":\s*"?([^",}]+)' % k.encode(), h).group(1).decode()
        return "%s_%s_t%s" % (g("task_suite"), g("safety_level"), g("task_id")), int(g("trial_id"))

    tasks, split = [], {}
    for rp, ms in groups.items():
        key = init_state(rp)
        if key in split_of:
            tasks.append((rp, ms, set()))
            split[rp] = split_of[key]
    print("[test_new] admitted %d, groups on old held-out init states %d (%d eps)"
          % (len(adm), len(tasks), sum(len(t[1]) for t in tasks)), flush=True)

    t0, rows, n_err = time.time(), [], 0
    with open(out / "stats.jsonl", "w") as f, open(out / "errors.jsonl", "w") as fe, Pool(a.workers) as p:
        for k, res in enumerate(p.imap_unordered(bci.process_group, tasks, chunksize=1)):
            for r in res:
                if "error" in r:
                    n_err += 1
                    fe.write(json.dumps(r) + "\n")
                    continue
                f.write(json.dumps(r) + "\n")
                rows.append(r)
            if k % 100 == 0:
                print("[test_new] group %d/%d ok %d err %d %.0fs" % (k, len(tasks), len(rows), n_err, time.time() - t0),
                      flush=True)

    # leakage guards: no file and no init state shared with cws_v1 train
    leak_json = sum(r["json"] in train_json for r in rows)
    leak_tt = sum((r["tcell"], r["trial"]) in train_tt for r in rows)
    if leak_json or leak_tt:
        raise SystemExit("[test_new] ABORT: %d files / %d init states overlap cws_v1 train" % (leak_json, leak_tt))

    root_split = {}
    for rp, ms, _ in tasks:
        for o in ms:
            root_split[o["audit_path"]] = split[rp]
    with open(out / "manifest.jsonl", "w") as f:
        for r in rows:
            r = dict(r, split=root_split[r["json"]], src="new_heldout", num_frames_raw=r["num_frames"])
            f.write(json.dumps(r, default=float) + "\n")

    man = [json.loads(l) for l in open(out / "manifest.jsonl")]
    rep = {}
    for s in ("test", "val"):
        rs = [r for r in man if r["split"] == s]
        roots = [r for r in rs if r["kind"] == "root"]
        rep[s] = dict(
            episodes=len(rs), dual=sum(bool(r["dual"]) for r in rs),
            quadrants=dict(collections.Counter(r["quadrant"] for r in rs)),
            roots=len(roots), dual_roots_ge129=dict(collections.Counter(
                r["quadrant"] for r in roots if r["dual"] and r["num_frames"] >= 129)),
            dual_roots_ge257=dict(collections.Counter(
                r["quadrant"] for r in roots if r["dual"] and r["num_frames"] >= 257)),
            init_states=len({(r["tcell"], r["trial"]) for r in rs}), cells=len({r["tcell"] for r in rs}))
    rep["errors"] = n_err
    json.dump(rep, open(out / "report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)
    print("[test_new] DONE %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
