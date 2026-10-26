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

"""SC-GRPO keyframe pool and proprio state model.

Keyframes: real chunk boundaries t0 (multiple of 8, t0 >= 8) of Wan-seen train groups
(roles cost_train / h_train / dev; never diag_* or fresh4) where a success or a safety event
happens within the next ``--horizon`` frames, plus a small random share. Per keyframe:
  anchor   frame 0 (dual 512x256, agent over wrist, policy view)
  memory   frames t0-3 .. t0 and the env actions that produced them (action[f-1])
  state    pi0.5 proprio at t0: eef_pos(3) + axis-angle(eef_quat)(3) + gripper_qpos(2)
Frame f's state: f = 0 initial observation, f >= 1 the observation after action f-1.

State model: s_{t+1} - s_t ~ linear in [a_t, s_t gripper, previous delta, 1] (least squares),
used to propagate the proprio state through imagined chunks. Reports 64-step open-loop error.
"""

from __future__ import annotations

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from toolkits.libero_safety.sam_alignment.cache_real import load_dual_frames
from toolkits.libero_safety.sam_alignment.schema import CHUNK, OUT_ROOT

TASKS = "/data_cf/cws_rm/frames_v1/tasks.json"
ROLES = ("cost_train", "h_train", "dev")


def quat2axisangle(q):
    """robosuite (x, y, z, w) quaternion -> axis-angle, as in LIBERO's policy state."""
    q = np.asarray(q, np.float64)
    w = np.clip(q[3], -1.0, 1.0)
    den = math.sqrt(max(1.0 - w * w, 0.0))
    if den < 1e-8:
        return np.zeros(3)
    return q[:3] * 2.0 * math.acos(w) / den


def state_of(obs):
    return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                           obs["robot0_gripper_qpos"]]).astype(np.float32)


def episode_states(e):
    """[num_frames, 8] proprio states aligned with the stitched frames."""
    rj = json.load(open(e["root_json"]))
    root = [state_of(rj["initial_observation"])] + [state_of(s["observation"]) for s in rj["trajectory"]]
    if e["kind"] == "root":
        st = root
    else:
        bt = int(e["branch_timestep"])
        bj = json.load(open(e["json"]))
        st = root[:bt] + [state_of(s["observation"]) for s in bj["trajectory"]]
    st = np.stack(st)[: e["num_frames"]]
    if len(st) != e["num_frames"]:
        raise ValueError(f"{e['uid']}: {len(st)} states for {e['num_frames']} frames")
    return st


def select(e, rng, horizon, cap, rest):
    with np.load(e["labels"]) as z:
        ev = (z["cost"] & z["valid_channel"]).any(1)
    n = e["num_frames"]
    fs = None
    if e["success"]:
        with np.load(e["labels"]) as z:
            s = np.flatnonzero(z["success"])
        fs = int(s[0]) + 1 if len(s) else None  # frame index after the successful action
    keep, kind = [], []
    for t0 in range(CHUNK, n - 1, CHUNK):
        succ = fs is not None and 0 < fs - t0 <= horizon
        evt = bool(ev[t0:t0 + horizon].any())
        if succ or evt or rng.random() < rest:
            keep.append(t0)
            kind.append(int(succ) + 2 * int(evt))
    if len(keep) > cap:
        idx = np.sort(rng.choice(len(keep), cap, replace=False))
        keep, kind = [keep[i] for i in idx], [kind[i] for i in idx]
    return keep, kind


def work(job):
    i, e, t0s = job
    st = episode_states(e)
    with np.load(e["labels"]) as z:
        act = z["action_env"].astype(np.float32)
    out = {"states": st, "actions": act, "rows": []}
    if t0s:
        fr = load_dual_frames(e)
        for t0 in t0s:
            mem_ids = [t0 - 3, t0 - 2, t0 - 1, t0]
            out["rows"].append({"anchor": fr[0], "memory": fr[mem_ids],
                                "mem_actions": act[[f - 1 for f in mem_ids]], "state": st[t0], "t0": t0})
    return i, out


def fit_state_model(trans):
    """trans: list of (states [n,8], actions [n-1,7]) -> W [features, 8], feature builder."""
    X, Y = [], []
    for st, act in trans:
        d = np.diff(st, axis=0)                       # [n-1, 8]
        prev = np.vstack([np.zeros((1, 8)), d[:-1]])  # previous delta
        X.append(np.hstack([act[: len(d)], st[:-1, 6:8], prev[:, :6], np.ones((len(d), 1))]))
        Y.append(d)
    X, Y = np.vstack(X), np.vstack(Y)
    W = np.linalg.lstsq(X, Y, rcond=None)[0]
    r2 = 1 - ((Y - X @ W) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    return W, r2


def rollout_state(W, s0, prev, actions):
    s, p, out = s0.copy(), prev.copy(), []
    for a in actions:
        x = np.concatenate([a, s[6:8], p[:6], [1.0]])
        d = x @ W
        s = s + d
        p = d
        out.append(s.copy())
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT_ROOT / "scgrpo" / "keyframes_v1"))
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--cap", type=int, default=6)
    ap.add_argument("--rest", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261026)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    reg = json.load(open(OUT_ROOT / "splits" / "split_registry.json"))
    tasks = json.load(open(TASKS))
    inst = {s: i for i, s in enumerate(tasks["instructions"])}
    eps = [json.loads(l) for l in open(OUT_ROOT / "labels" / "episodes.jsonl")]
    eps = [e for e in eps if reg["episode_role"].get(e["uid"]) in ROLES and e["label_source"] == "grader_v1"
           and e["dual"]]
    if args.limit:
        eps = eps[: args.limit]
    jobs, kinds = [], {}
    for i, e in enumerate(eps):
        t0s, k = select(e, rng, args.horizon, args.cap, args.rest)
        jobs.append((i, e, t0s))
        kinds[i] = k
    n_kf = sum(len(j[2]) for j in jobs)
    print(f"{len(eps)} episodes, {n_kf} keyframes", flush=True)

    anchor = np.lib.format.open_memmap(out / "anchor.npy", "w+", np.uint8, (n_kf, 512, 256, 3))
    memory = np.lib.format.open_memmap(out / "memory.npy", "w+", np.uint8, (n_kf, 4, 512, 256, 3))
    meta = {k: [] for k in ("uid", "group", "tcell", "suite", "task_id", "instruction", "t0", "kind",
                            "quadrant", "role")}
    mem_actions, states, trans = np.zeros((n_kf, 4, 7), np.float32), np.zeros((n_kf, 8), np.float32), []
    row = 0
    with Pool(args.workers) as pool:
        for n_done, (i, res) in enumerate(pool.imap_unordered(work, jobs, chunksize=4)):
            e = eps[i]
            trans.append((res["states"], res["actions"]))
            for r, k in zip(res["rows"], kinds[i]):
                anchor[row], memory[row] = r["anchor"], r["memory"]
                mem_actions[row], states[row] = r["mem_actions"], r["state"]
                for key, v in (("uid", e["uid"]), ("group", e["group"]), ("tcell", e["tcell"]),
                               ("suite", e["suite"]), ("task_id", inst[tasks["tcell_instruction"][e["tcell"]]]),
                               ("instruction", tasks["tcell_instruction"][e["tcell"]]), ("t0", r["t0"]),
                               ("kind", k), ("quadrant", e["actual_quadrant"]),
                               ("role", reg["episode_role"][e["uid"]])):
                    meta[key].append(v)
                row += 1
            if (n_done + 1) % 500 == 0:
                print(f"{n_done + 1}/{len(jobs)} episodes, {row} keyframes", flush=True)
    anchor.flush()
    memory.flush()
    np.savez(out / "meta.npz", mem_actions=mem_actions, states=states,
             **{k: np.array(v) for k, v in meta.items()})

    # state model: fit on 90% of episodes, 64-step open-loop error on the rest
    perm = rng.permutation(len(trans))
    cut = int(0.9 * len(trans))
    W, r2 = fit_state_model([trans[j] for j in perm[:cut]])
    errs = []
    for j in perm[cut:]:
        st, act = trans[j]
        for t0 in range(CHUNK, len(st) - 65, 32):
            pred = rollout_state(W, st[t0], st[t0] - st[t0 - 1], act[t0:t0 + 64])
            errs.append(np.abs(pred - st[t0 + 1:t0 + 65]))
    errs = np.stack(errs)  # [n, 64, 8]
    rep = {"keyframes": int(row), "episodes": len(eps),
           "kind_counts": {k: int((np.array(meta["kind"]) == v).sum()) for k, v in
                           (("rest", 0), ("success", 1), ("event", 2), ("both", 3))},
           "per_suite": {s: int((np.array(meta["suite"]) == s).sum()) for s in sorted(set(meta["suite"]))},
           "state_model_r2": r2.round(3).tolist(),
           "open_loop_mae_pos_m": {str(h): errs[:, h - 1, :3].mean().round(4).item() for h in (8, 32, 64)},
           "open_loop_mae_rot_rad": {str(h): errs[:, h - 1, 3:6].mean().round(4).item() for h in (8, 32, 64)},
           "open_loop_mae_grip_m": {str(h): errs[:, h - 1, 6:8].mean().round(5).item() for h in (8, 32, 64)},
           "state_std_norm": [0.139, 0.178, 0.354, 0.468, 1.558, 0.211, 0.013, 0.013]}
    np.save(out / "state_model_W.npy", W)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
