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

"""Diagnose step-2 episodes that succeed in the simulator but not inside Wan.

Per episode (real success, Wan closed-loop failure):
  state  : estimator driven by the REAL actions vs the recorded proprio state
  wiring : pi0.5 chunk at frame 0 (real frame + real state) vs the recorded actions
  replay : Wan driven by the REAL actions (open loop) + RM on its agent frames.
           RM fires here -> RM works on Wan frames; it does not -> RM / WM issue.
  loop   : pi0.5 closed loop inside Wan (as in step2_wan), first --loop_steps steps
Writes <out>/<tag>.mp4 (columns real | replay | loop; agent over wrist) and
<out>/<tag>_sheet.png (every 16th frame), plus <out>/diag.json.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from toolkits.libero_safety.wan_ac_sft.sampler import NULL_ACTION, scale_pose
from toolkits.libero_safety.wan_ac_sft.step2_wan import (StateEstimator, load_rm, load_train_trajectories, obs_seq,
                                                          rm_probs)
from toolkits.libero_safety.wan_ac_sft.wrist_sens import load_pi05, predict, state_of


def wan_chunk(pipe, anchor, memory, mem_act, chunk_act, seed, steps, dev):
    a = np.concatenate([NULL_ACTION[None], mem_act, chunk_act], 0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = pipe(seed=seed, tiled=False, input_image=[anchor], input_image4=[memory],
                   action=torch.from_numpy(a).unsqueeze(0).to(dev), height=512, width=256, num_frames=13,
                   num_inference_steps=steps, cfg_scale=1.0, progress_bar_cmd=lambda x: x, batch_size=1)
    return [np.asarray(im, np.uint8) for im in out[0][5:]]


def label(img, text):
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text((4, 4), text, fill=(255, 255, 0))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", required=True, help="comma-separated audit JSON paths")
    ap.add_argument("--dit", required=True)
    ap.add_argument("--vae", default="/mnt/nfs/fjj/wan_ac_sft/ckpt/RLinf-Wan-LIBERO-Spatial/Wan2.2_VAE.pth")
    ap.add_argument("--rm", default="/data_cf/cws_rm/runs/rm_v1_agent/best.pt")
    ap.add_argument("--index", default="/data_cf/wan_ac_sft/index/cws_v1")
    ap.add_argument("--loop_steps", type=int, default=160)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--pose_scale", type=float, default=8.0)
    ap.add_argument("--pose_clip", type=float, default=4.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    est = StateEstimator().fit(load_train_trajectories(args.index, 160, 0))
    rm = load_rm(args.rm, dev)
    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline
    from toolkits.libero_safety.wan_ac_sft.dual_view import enable_dual
    pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=dev, model_configs=[
        ModelConfig(path=args.dit, offload_device="cpu"), ModelConfig(path=args.vae, offload_device="cpu")])
    pipe.dit.to(dev)
    pipe.vae.to(dev)
    enable_dual(pipe)
    policy = load_pi05(dev)

    report = []
    for jp in args.episodes.split(","):
        j = json.load(open(jp))
        meta, oc = j["episode"], j["outcome"]
        tag = f"{meta['task_suite']}_{meta['safety_level']}_t{meta['task_id']}_trial{meta['trial_id']}"
        z = np.load(os.path.join(os.path.dirname(jp), "visual", os.path.basename(jp).replace(".json", ".npz")))
        ag, wr = z["agentview_image"], z["robot0_eye_in_hand_image"]
        acts = np.array([s["action"] for s in j["trajectory"]], np.float32)
        obs = obs_seq(j)
        prompt = meta["instruction"]
        n_real = len(acts)
        rec = {"tag": tag, "json": jp, "real_success_step": oc.get("first_raw_task_success_timestep"), "len": n_real}

        # state: estimator on real actions vs recorded proprio
        est.reset(obs[0])
        st_err = {}
        for t, a in enumerate(acts, 1):
            est.step(a)
            if t in (1, 8, 16, 32, 64, 128):
                d = est.state() - state_of(obs[t])
                st_err[str(t)] = {"pos_cm": float(100 * np.linalg.norm(d[:3])), "rot": float(np.linalg.norm(d[3:6])),
                                  "grip": float(np.abs(d[6:]).max())}
        rec["state_err"] = st_err
        rec["state0"] = state_of(obs[0]).round(4).tolist()

        # wiring: pi0.5 at frame 0 vs the recorded chunk
        p0 = predict(policy, ag[:1], wr[:1], state_of(obs[0])[None], prompt, 0)[0]
        rec["wiring"] = {"pred_chunk0": p0.round(3).tolist(), "rec_chunk0": acts[:8].round(3).tolist(),
                         "pose_l1": float(np.abs(p0[:, :6] - acts[:8, :6]).mean()),
                         "grip_agree": float((np.sign(p0[:, 6]) == np.sign(acts[:8, 6])).mean())}

        # replay: Wan driven by the real actions
        anchor = Image.fromarray(np.concatenate([ag[0], wr[0]], 0))
        memory, mem_act, rep = [anchor] * 4, np.repeat(NULL_ACTION[None], 4, 0), []
        sc_all = scale_pose(acts, args.pose_scale, args.pose_clip)
        for c in range((n_real + 7) // 8):
            ch = sc_all[8 * c: 8 * c + 8]
            if len(ch) < 8:
                ch = np.concatenate([ch, np.repeat(NULL_ACTION[None], 8 - len(ch), 0)], 0)
            new = wan_chunk(pipe, anchor, memory, mem_act, ch, 1000 * c, args.steps, dev)
            rep += new
            memory, mem_act = [Image.fromarray(x) for x in new[-4:]], ch[-4:]
        rep = np.stack(rep[:n_real])
        p_rep = rm_probs(rm, rep[:, :256], prompt, dev)
        p_real = rm_probs(rm, ag, prompt, dev)

        # loop: pi0.5 closed loop in Wan
        est.reset(obs[0])
        cur_a, cur_w = ag[0], wr[0]
        memory, mem_act, loop, loop_acts, loop_states = [anchor] * 4, np.repeat(NULL_ACTION[None], 4, 0), [], [], []
        for c in range(args.loop_steps // 8):
            loop_states.append(est.state().round(4).tolist())
            act = predict(policy, cur_a[None], cur_w[None], est.state()[None], prompt, c + 1)[0]
            sc = scale_pose(act.astype(np.float32), args.pose_scale, args.pose_clip)
            new = wan_chunk(pipe, anchor, memory, mem_act, sc, 1000 * (c + 1), args.steps, dev)
            for x in act:
                est.step(x)
            loop += new
            loop_acts += act.tolist()
            memory, mem_act = [Image.fromarray(x) for x in new[-4:]], sc[-4:]
            cur_a, cur_w = new[-1][:256], new[-1][256:]
        loop = np.stack(loop)
        p_loop = rm_probs(rm, loop[:, :256], prompt, dev)
        la = np.array(loop_acts)
        rec["loop"] = {"p_max": float(p_loop.max()), "grip_close_step": int(np.argmax(la[:, 6] > 0)) if (la[:, 6] > 0).any() else None,
                       "real_grip_close_step": int(np.argmax(acts[:, 6] > 0)) if (acts[:, 6] > 0).any() else None,
                       "act_mean_abs": np.abs(la[:, :6]).mean(0).round(4).tolist(),
                       "real_act_mean_abs": np.abs(acts[: len(la), :6]).mean(0).round(4).tolist(),
                       "est_states_per_chunk": loop_states[:6]}
        rec["rm"] = {"real_max": float(p_real.max()), "replay_max": float(p_rep.max()),
                     "replay_last8": p_rep[-8:].round(3).tolist(), "real_last8": p_real[-8:].round(3).tolist()}

        # video: real | replay | loop (agent over wrist)
        T = max(n_real + 1, len(loop) + 1)
        real_v = np.concatenate([ag, wr], 1)
        blank = np.zeros_like(real_v[0])
        frames = []
        for t in range(T):
            r = real_v[t] if t < len(real_v) else blank
            p = (np.concatenate([ag[0], wr[0]], 0) if t == 0 else rep[t - 1]) if t <= len(rep) else blank
            l = (np.concatenate([ag[0], wr[0]], 0) if t == 0 else loop[t - 1]) if t <= len(loop) else blank
            pr = p_real[t] if t < len(p_real) else float("nan")
            pp = p_rep[t - 1] if 0 < t <= len(p_rep) else float("nan")
            pl = p_loop[t - 1] if 0 < t <= len(p_loop) else float("nan")
            frames.append(np.concatenate([label(r, f"real t{t} rm {pr:.2f}"), label(p, f"replay rm {pp:.2f}"),
                                          label(l, f"loop rm {pl:.2f}")], 1))
        imageio.mimsave(out / f"{tag}.mp4", frames, fps=10)
        idx = list(range(0, T, 16))[:12]
        sheet = np.concatenate([np.concatenate([frames[i] for i in idx[k:k + 4]], 1) for k in range(0, len(idx), 4)
                                if len(idx[k:k + 4]) == 4], 0)
        Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2)).save(out / f"{tag}_sheet.png")
        report.append(rec)
        print(f"[diag] {tag} real_succ@{rec['real_success_step']} rm real/replay/loop max "
              f"{p_real.max():.3f}/{p_rep.max():.3f}/{p_loop.max():.3f} | wiring pose_l1 {rec['wiring']['pose_l1']:.4f} "
              f"grip {rec['wiring']['grip_agree']:.2f} | state err@64 {st_err.get('64')} | grip close real "
              f"{rec['loop']['real_grip_close_step']} loop {rec['loop']['grip_close_step']}", flush=True)
        json.dump(report, open(out / "diag.json", "w"), indent=1)


if __name__ == "__main__":
    main()
