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

"""pi0.5 action sensitivity to Wan-generated wrist frames (teacher-forced, open loop).

For each held-out dual root, Wan (dual view) rolls out ``--chunks`` x 8 frames from
frame 0 with the episode's own actions. At probe frames f = 8c (c in ``--probe``)
pi0.5 predicts an 8-step action chunk from the REAL agent frame + REAL proprio
state and one of these wrist inputs:
  real   : recorded wrist frame                     (reference)
  floor  : recorded wrist frame, another noise seed (sampling noise floor)
  wan    : Wan-generated wrist frame
  mask   : no wrist (RLINF_PI05_WRIST_MODE=mask, upper bound of a useless wrist)
  both   : Wan agent + Wan wrist (full imagination, for reference)
Deviation d(cond) = mean L2 over the 8 actions' pose dims (0-5) vs ``real`` with the
same seed, plus gripper sign agreement. wan_over_mask = d_wan / d_mask (same seed):
0 = Wan wrist is as good as the recorded one, 1 = as bad as no wrist. ``floor`` is the
seed-to-seed spread for scale. Open loop only: per-step effects compound in closed loop.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from toolkits.libero_safety.wan_ac_sft.sampler import NULL_ACTION, load_frames, load_manifest, scale_pose

PI05 = "/mnt/nfs/fjj/Code/rlinf-safety-runtime/checkpoints/pi05_libero_safety_rlinf"
CONDS = ("real", "floor", "wan", "mask", "both")


def quat2axisangle(q):
    q = np.array(q, np.float64)
    q[3] = min(max(q[3], -1.0), 1.0)
    den = math.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return q[:3] * 2.0 * math.acos(q[3]) / den


def state_of(obs):
    return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                           obs["robot0_gripper_qpos"]]).astype(np.float32)


def load_pi05(dev):
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.openpi import get_model
    cfg = OmegaConf.create({
        "model_path": PI05,
        "openpi": {"task": "eval", "config_name": "pi05_libero", "num_images_in_input": 2, "num_steps": 5,
                   "noise_method": "flow_sde", "noise_level": 0.3, "action_chunk": 8, "action_env_dim": 7},
        "openpi_data": {"norm_stats_path": f"{PI05}/assets/lerobot/norm_stats.json"}})
    return get_model(cfg).to(dev).eval()


@torch.no_grad()
def predict(policy, agent, wrist, state, prompt, seed, wrist_mode="real"):
    """agent / wrist uint8 [B,256,256,3], state [B,8] -> actions [B,8,7]."""
    os.environ["RLINF_PI05_WRIST_MODE"] = wrist_mode
    torch.manual_seed(seed)
    b = len(agent)
    obs = {"main_images": torch.from_numpy(np.ascontiguousarray(agent)),
           "wrist_images": torch.from_numpy(np.ascontiguousarray(wrist)),
           "states": torch.from_numpy(np.ascontiguousarray(state)),
           "task_descriptions": [prompt] * b, "extra_view_images": None}
    act, _ = policy.predict_action_batch(obs, mode="eval", compute_values=False)
    os.environ["RLINF_PI05_WRIST_MODE"] = "real"
    act = act.detach().float().cpu().numpy() if torch.is_tensor(act) else np.asarray(act, np.float32)
    return act.reshape(b, -1, 7)[:, :8]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", default="/data_cf/wan_ac_sft/index/cws_test_new")
    ap.add_argument("--split", default="test")
    ap.add_argument("--dit", required=True)
    ap.add_argument("--vae", default="/mnt/nfs/fjj/wan_ac_sft/ckpt/RLinf-Wan-LIBERO-Spatial/Wan2.2_VAE.pth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--probe", default="1,4,8,16")
    ap.add_argument("--max_roots", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--pose_scale", type=float, default=8.0)
    ap.add_argument("--pose_clip", type=float, default=4.0)
    ap.add_argument("--orientation", default="stored", choices=["", "stored", "rot180"],
                    help="stored = NPZ frames as saved (the auditor already rotates to the policy view); '' = auto")
    args = ap.parse_args()
    probes = [int(x) for x in args.probe.split(",")]
    T = args.chunks * 8
    dev = "cuda"

    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline
    from toolkits.libero_safety.wan_ac_sft.dual_view import enable_dual
    pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=dev, model_configs=[
        ModelConfig(path=args.dit, offload_device="cpu"), ModelConfig(path=args.vae, offload_device="cpu")])
    pipe.dit.to(dev)
    pipe.vae.to(dev)
    enable_dual(pipe)
    policy = load_pi05(dev)

    rows = [r for r in load_manifest(args.index, args.split)
            if r["kind"] == "root" and r.get("dual") and r["num_frames"] >= T + 9]  # 8 recorded actions after the last probe
    rows.sort(key=lambda r: r["idx"])
    if args.max_roots:
        rows = rows[:: max(1, len(rows) // args.max_roots)][: args.max_roots]

    def rollout(f0, acts, seed=0):
        anchor = Image.fromarray(f0)
        memory, mem_act, frames = [anchor] * 4, np.repeat(NULL_ACTION[None], 4, 0), []
        for c in range(args.chunks):
            chunk_act = acts[8 * c: 8 * c + 8]
            a = np.concatenate([NULL_ACTION[None], mem_act, chunk_act], 0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = pipe(seed=seed + 1000 * c, tiled=False, input_image=[anchor], input_image4=[memory],
                           action=torch.from_numpy(a).unsqueeze(0).to(dev), height=512, width=256,
                           num_frames=13, num_inference_steps=args.steps, cfg_scale=1.0,
                           progress_bar_cmd=lambda x: x, batch_size=1)
            new = [np.asarray(im, np.uint8) for im in out[0][5:]]
            frames += new
            memory, mem_act = [Image.fromarray(x) for x in new[-4:]], chunk_act[-4:]
        return np.stack(frames)  # frames[i] = frame i + 1

    # orientation check on the first roots: as stored vs rotated 180, against recorded actions
    def states_of(r, fids):
        j = json.load(open(r["json"]))
        obs = [j["initial_observation"]] + [s["observation"] for s in j["trajectory"]]
        return np.stack([state_of(obs[f]) for f in fids]), j["episode"]["instruction"], \
            np.array([s["action"] for s in j["trajectory"]], np.float32)

    fids = [8 * c for c in probes]
    ori_err = {"stored": [], "rot180": []}
    for r in rows[:6]:
        ag = load_frames(r["agent_segments"], fids, "agentview_image")
        wr = load_frames(r["wrist_segments"], fids, "robot0_eye_in_hand_image")
        st, prompt, acts = states_of(r, fids)
        ref = np.stack([acts[f: f + 8] for f in fids])
        for name, tf in (("stored", lambda x: x), ("rot180", lambda x: x[:, ::-1, ::-1])):
            p = predict(policy, tf(ag), tf(wr), st, prompt, 0)
            ori_err[name].append(float(np.abs(p[..., :6] - ref[..., :6]).mean()))
    ori = args.orientation or min(ori_err, key=lambda k: np.mean(ori_err[k]))
    flip = (lambda x: x[:, ::-1, ::-1]) if ori == "rot180" else (lambda x: x)
    print(f"[sens] orientation check |pred-recorded| pose: "
          + ", ".join(f"{k} {np.mean(v):.4f}" for k, v in ori_err.items()) + f" -> {ori}", flush=True)

    recs, t0 = [], time.time()
    for k, r in enumerate(rows):
        ag = load_frames(r["agent_segments"], list(range(T + 1)), "agentview_image")
        wr = load_frames(r["wrist_segments"], list(range(T + 1)), "robot0_eye_in_hand_image")
        st, prompt, acts = states_of(r, fids)
        gen = rollout(np.concatenate([ag[0], wr[0]], 0),
                      scale_pose(acts[:T], args.pose_scale, args.pose_clip))
        gag, gwr = gen[:, :256], gen[:, 256:]
        A, W = flip(ag[fids]), flip(wr[fids])
        GA, GW = flip(gag[[f - 1 for f in fids]]), flip(gwr[[f - 1 for f in fids]])
        out = {"real": predict(policy, A, W, st, prompt, 0),
               "floor": predict(policy, A, W, st, prompt, 1),
               "wan": predict(policy, A, GW, st, prompt, 0),
               "mask": predict(policy, A, W, st, prompt, 0, wrist_mode="mask"),
               "both": predict(policy, GA, GW, st, prompt, 0)}
        rec = {"idx": r["idx"], "tcell": r["tcell"], "quadrant": r["quadrant"], "probe": probes, "d": {}, "grip": {}}
        for c in CONDS[1:]:
            rec["d"][c] = np.linalg.norm(out[c][..., :6] - out["real"][..., :6], axis=-1).mean(-1).tolist()
            rec["grip"][c] = (np.sign(out[c][..., 6]) == np.sign(out["real"][..., 6])).mean(-1).tolist()
        rec["d_rec"] = {c: np.linalg.norm(out[c][..., :6] - np.stack([acts[f: f + 8] for f in fids])[..., :6],
                                          axis=-1).mean(-1).tolist() for c in CONDS}
        recs.append(rec)
        print(f"[sens] {k + 1}/{len(rows)} {time.time() - t0:.0f}s {r['tcell']} {r['quadrant']} "
              + " ".join(f"{c}={rec['d'][c][-1]:.3f}" for c in CONDS[1:]), flush=True)

    summ = {"n": len(recs), "probe_chunks": probes, "orientation": ori, "orientation_err": ori_err}
    for c in CONDS[1:]:
        summ[f"d_{c}"] = np.mean([x["d"][c] for x in recs], 0).tolist()
        summ[f"grip_{c}"] = np.mean([x["grip"][c] for x in recs], 0).tolist()
    for c in CONDS:
        summ[f"d_rec_{c}"] = np.mean([x["d_rec"][c] for x in recs], 0).tolist()
    # same-seed ratio: how far the Wan wrist moves the action relative to removing the wrist
    # (both vs the real-wrist prediction with the same noise); 0 = no effect, 1 = as bad as no wrist
    dw, dm = (np.array(summ[f"d_{c}"]) for c in ("wan", "mask"))
    summ["wan_over_mask"] = (dw / np.maximum(dm, 1e-9)).tolist()
    summ["wan_over_mask_per_root_median"] = np.median(
        [np.array(x["d"]["wan"]) / np.maximum(np.array(x["d"]["mask"]), 1e-9) for x in recs], 0).tolist()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summ, "args": vars(args), "roots": recs}, open(args.out, "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
