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

"""Policy-in-the-loop gate, Wan side: pi0.5 closed loop inside the dual-view Wan.

For every real step-2 episode (pi0.5, 8-action chunks, LIBERO-Safety) the same
policy runs inside Wan from the episode's frame 0 (agent + wrist) and initial
proprio state: pi0.5 sees the latest generated frames, outputs 8 actions, Wan
generates the next 8 frames, until success or 512 steps.

* proprio state: Wan predicts pixels only, so eef pos / rot / gripper are integrated
  from the actions with a linear model (current + previous action + previous delta)
  fitted on cws_v1 train trajectories; its open-loop drift is reported.
* success: agent-view RM (``--rm``), success when p > thr for ``--consec`` frames in a
  row. The same rule is applied to the real frames, so RM error and WM error separate.
Reports per cell: real SR (simulator), RM-on-real SR, Wan SR, |dSR|.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

from toolkits.libero_safety.wan_ac_sft.sampler import NULL_ACTION, load_manifest, scale_pose
from toolkits.libero_safety.wan_ac_sft.wrist_sens import load_pi05, predict, quat2axisangle

RM_FILE = "/mnt/nfs/fjj/Code/CodeWorld-Safe/rlinf/models/embodiment/reward/cws_reward_model.py"


def obs_seq(j):
    return [j["initial_observation"]] + [s["observation"] for s in j["trajectory"]]


class StateEstimator:
    """Linear one-step model for eef pos / rotation (world rotvec) / gripper qpos."""

    def fit(self, episodes):
        Xp, Yp, Xr, Yr, Xg, Yg = [], [], [], [], [], []
        for acts, obs in episodes:
            pos = np.array([o["robot0_eef_pos"] for o in obs])
            R = Rot.from_quat(np.array([o["robot0_eef_quat"] for o in obs]))
            q = np.array([o["robot0_gripper_qpos"] for o in obs])
            dp = np.diff(pos, axis=0)
            dr = (R[1:] * R[:-1].inv()).as_rotvec()
            for t in range(1, len(acts)):
                Xp.append(np.r_[acts[t, :3], acts[t - 1, :3], dp[t - 1], 1.0]); Yp.append(dp[t])
                Xr.append(np.r_[acts[t, 3:6], acts[t - 1, 3:6], dr[t - 1], 1.0]); Yr.append(dr[t])
            for t in range(len(acts)):
                Xg.append(np.r_[q[t], acts[t, 6], 1.0]); Yg.append(q[t + 1])
        self.Wp = np.linalg.lstsq(np.array(Xp), np.array(Yp), rcond=None)[0]
        self.Wr = np.linalg.lstsq(np.array(Xr), np.array(Yr), rcond=None)[0]
        self.Wg = np.linalg.lstsq(np.array(Xg), np.array(Yg), rcond=None)[0]
        return self

    def reset(self, obs0):
        self.pos = np.array(obs0["robot0_eef_pos"], np.float64)
        self.quat = np.array(obs0["robot0_eef_quat"], np.float64)
        self.q = np.array(obs0["robot0_gripper_qpos"], np.float64)
        self.a_prev = np.zeros(7)
        self.dp_prev = np.zeros(3)
        self.dr_prev = np.zeros(3)

    def step(self, a):
        a = np.asarray(a, np.float64)
        dp = np.r_[a[:3], self.a_prev[:3], self.dp_prev, 1.0] @ self.Wp
        dr = np.r_[a[3:6], self.a_prev[3:6], self.dr_prev, 1.0] @ self.Wr
        self.q = np.r_[self.q, a[6], 1.0] @ self.Wg
        self.pos = self.pos + dp
        nq = (Rot.from_rotvec(dr) * Rot.from_quat(self.quat)).as_quat()
        self.quat = nq if np.dot(nq, self.quat) >= 0 else -nq  # keep the sign continuous
        self.a_prev, self.dp_prev, self.dr_prev = a, dp, dr

    def state(self):
        return np.concatenate([self.pos, quat2axisangle(self.quat), self.q]).astype(np.float32)


def load_train_trajectories(index, n, seed):
    rows = [r for r in load_manifest(index, "train") if r["kind"] == "root" and r.get("src") == "new"]
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows[:n]:
        j = json.load(open(r["json"]))
        out.append((np.array([s["action"] for s in j["trajectory"]], np.float64), obs_seq(j)))
    return out


def drift_report(est, episodes, horizons=(8, 64, 128, 256, 512)):
    err = collections.defaultdict(list)
    frozen = collections.defaultdict(list)
    for acts, obs in episodes:
        est.reset(obs[0])
        p0 = np.array(obs[0]["robot0_eef_pos"])
        for t, a in enumerate(acts, 1):
            est.step(a)
            if t in horizons:
                p = np.array(obs[t]["robot0_eef_pos"])
                err[t].append(np.linalg.norm(est.pos - p))
                frozen[t].append(np.linalg.norm(p0 - p))
    return {str(h): {"pos_err_cm": 100 * float(np.median(err[h])), "frozen_err_cm": 100 * float(np.median(frozen[h])),
                     "n": len(err[h])} for h in horizons if err[h]}


def load_rm(path, dev):
    spec = importlib.util.spec_from_file_location("cws_reward_model", RM_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CWSRewardModel.from_checkpoint(path).to(dev).eval()


@torch.no_grad()
def rm_probs(rm, frames, prompt, dev):
    """agent frames uint8 [N,256,256,3] -> success prob [N]."""
    out = []
    for a in range(0, len(frames), 128):
        x = torch.from_numpy(np.ascontiguousarray(frames[a:a + 128])).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1
        out.append(rm.predict_rew(x, [prompt] * len(x)).float().flatten().cpu().numpy())
    return np.concatenate(out)


def first_success(p, thr, consec):
    run = 0
    for i, x in enumerate(p):
        run = run + 1 if x > thr else 0
        if run >= consec:
            return i
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--real", default="/data_cf/cws/step2_real")
    ap.add_argument("--dit", required=True)
    ap.add_argument("--vae", default="/mnt/nfs/fjj/wan_ac_sft/ckpt/RLinf-Wan-LIBERO-Spatial/Wan2.2_VAE.pth")
    ap.add_argument("--rm", default="/data_cf/cws_rm/runs/rm_v1_agent/best.pt")
    ap.add_argument("--rm_thr", type=float, default=0.0, help="0 = newval best_thr from the RM final.json")
    ap.add_argument("--consec", type=int, default=3)
    ap.add_argument("--index", default="/data_cf/wan_ac_sft/index/cws_v1")
    ap.add_argument("--fit_eps", type=int, default=160)
    ap.add_argument("--max_steps", type=int, default=512)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--pose_scale", type=float, default=8.0)
    ap.add_argument("--pose_clip", type=float, default=4.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--report_only", action="store_true", help="only aggregate episodes_*.jsonl in --out")
    args = ap.parse_args()
    dev = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        return report(out, args)

    # proprio estimator + drift on held-out training trajectories
    trajs = load_train_trajectories(args.index, args.fit_eps + 40, 0)
    est = StateEstimator().fit(trajs[: args.fit_eps])
    drift = drift_report(est, trajs[args.fit_eps:])
    print("[s2w] estimator drift (median eef pos error, cm):", json.dumps(drift), flush=True)

    thr = args.rm_thr or json.load(open(Path(args.rm).parent / "final.json"))["val"]["newval"]["frame"]["best_thr"]
    rm = load_rm(args.rm, dev)

    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline
    from toolkits.libero_safety.wan_ac_sft.dual_view import enable_dual
    pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=dev, model_configs=[
        ModelConfig(path=args.dit, offload_device="cpu"), ModelConfig(path=args.vae, offload_device="cpu")])
    pipe.dit.to(dev)
    pipe.vae.to(dev)
    enable_dual(pipe)
    policy = load_pi05(dev)

    eps = sorted(str(p) for p in Path(args.real).rglob("*.json") if "libero_safety_audits" in str(p))
    eps = eps[args.shard:: args.nshard]
    ef = out / f"episodes_{args.shard}.jsonl"
    done = {json.loads(l)["json"] for l in open(ef)} if ef.exists() else set()
    fo = open(ef, "a")
    json.dump({"estimator_drift": drift, "rm": args.rm, "dit": args.dit}, open(out / "meta.json", "w"), indent=1)
    t0 = time.time()
    for k, jp in enumerate(eps):
        if jp in done:
            continue
        j = json.load(open(jp))
        meta, oc = j["episode"], j["outcome"]
        z = np.load(os.path.join(os.path.dirname(jp), "visual", os.path.basename(jp).replace(".json", ".npz")))
        ag, wr = z["agentview_image"], z["robot0_eye_in_hand_image"]
        prompt = meta["instruction"]
        p_real = rm_probs(rm, ag, prompt, dev)

        est.reset(j["initial_observation"])
        cur_a, cur_w = ag[0], wr[0]
        anchor = Image.fromarray(np.concatenate([ag[0], wr[0]], 0))
        memory, mem_act = [anchor] * 4, np.repeat(NULL_ACTION[None], 4, 0)
        p_wan, steps, seed = [], 0, 0
        while steps < args.max_steps:
            act = predict(policy, cur_a[None], cur_w[None], est.state()[None], prompt, seed)[0]  # [8,7]
            seed += 1
            sc = scale_pose(act.astype(np.float32), args.pose_scale, args.pose_clip)
            a = np.concatenate([NULL_ACTION[None], mem_act, sc], 0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gen = pipe(seed=1000 * seed, tiled=False, input_image=[anchor], input_image4=[memory],
                           action=torch.from_numpy(a).unsqueeze(0).to(dev), height=512, width=256, num_frames=13,
                           num_inference_steps=args.steps, cfg_scale=1.0, progress_bar_cmd=lambda x: x, batch_size=1)
            new = [np.asarray(im, np.uint8) for im in gen[0][5:]]
            for x in act:
                est.step(x)
            memory, mem_act = [Image.fromarray(x) for x in new[-4:]], sc[-4:]
            cur_a, cur_w = new[-1][:256], new[-1][256:]
            p_wan += rm_probs(rm, np.stack([x[:256] for x in new]), prompt, dev).tolist()
            steps += 8
            # no early stop: the full probability trace is kept so any success rule can be applied post hoc
        fs_w, fs_r = first_success(p_wan, thr, args.consec), first_success(p_real, thr, args.consec)
        rec = {"json": jp, "cell": f"{meta['task_suite']}_{meta['safety_level']}", "task": meta["task_id"],
               "trial": meta["trial_id"], "real_success": bool(oc["raw_task_success"]),
               "real_success_step": oc.get("first_raw_task_success_timestep"), "real_unsafe": bool(oc["unsafe"]),
               "rm_real_success": fs_r is not None, "rm_real_step": fs_r,
               "wan_success": fs_w is not None, "wan_step": None if fs_w is None else fs_w + 1, "wan_steps_run": steps,
               "p_real": np.round(p_real, 4).tolist(), "p_wan": np.round(np.array(p_wan), 4).tolist()}
        fo.write(json.dumps(rec) + "\n")
        fo.flush()
        print(f"[s2w] {k + 1}/{len(eps)} {time.time() - t0:.0f}s {rec['cell']} t{rec['task']} trial{rec['trial']} "
              f"real={rec['real_success']}@{rec['real_success_step']} rm_real={rec['rm_real_success']} "
              f"wan={rec['wan_success']}@{rec['wan_step']}", flush=True)
    fo.close()


def report(out, args):
    """Same success rule on real and Wan traces. Rules are swept; the one that best
    reproduces the simulator label on REAL episodes is chosen (calibrated on real only)."""
    recs = [json.loads(l) for f in sorted(out.glob("episodes_*.jsonl")) for l in open(f)]
    rules = [(t, c) for t in (0.5, 0.7, 0.9, 0.95, 0.99, 0.995) for c in (1, 2, 3)]

    def fires(p, t, c):
        return first_success(p, t, c) is not None

    cal = {f"{t}/{c}": float(np.mean([fires(r["p_real"], t, c) == r["real_success"] for r in recs])) for t, c in rules}
    best = max(rules, key=lambda tc: cal[f"{tc[0]}/{tc[1]}"])
    by = collections.defaultdict(list)
    for r in recs:
        by[r["cell"]].append(r)
        by["ALL"].append(r)
    meta = json.load(open(out / "meta.json")) if (out / "meta.json").exists() else {}
    rep = {**meta, "n": len(recs), "rule_calibration_real_acc": cal, "rule": {"thr": best[0], "consec": best[1]},
           "cells": {}}
    for c, rs in sorted(by.items()):
        real = np.array([r["real_success"] for r in rs], float)
        rmr = np.array([fires(r["p_real"], *best) for r in rs], float)
        wan = np.array([fires(r["p_wan"], *best) for r in rs], float)
        rep["cells"][c] = {"n": len(rs), "real_sr": real.mean(), "rm_real_sr": rmr.mean(), "wan_sr": wan.mean(),
                           "abs_dsr_vs_real": abs(wan.mean() - real.mean()),
                           "abs_dsr_vs_rm_real": abs(wan.mean() - rmr.mean()),
                           "episode_agree_vs_real": float((wan == real).mean())}
    json.dump(rep, open(out / "report.json", "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("n", "rule")}, indent=1), flush=True)
    print(json.dumps(rep["cells"], indent=1), flush=True)


if __name__ == "__main__":
    main()
