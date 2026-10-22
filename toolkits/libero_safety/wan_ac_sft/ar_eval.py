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

"""Multi-chunk autoregressive evaluation of a Wan AC world model.

From each held-out root's frame 0, the model rolls out ``--chunks`` x 8 frames
chunk by chunk, exactly like the WM backend: anchor = frame 0, memory = the last
4 generated frames (frame 0 repeated for the first chunk), actions = the 4 that
produced the memory frames + the 8 chunk actions. Rollouts:
  real_s{k}  : the episode's own actions, seeds 0..S-1
  shuf_s{k}  : actions of a root from a different task (same length), seeds 0..1
Per chunk index (time), LPIPS is reported for
  err_real   : real_s0 vs ground truth          err_shuf : shuf_s0 vs ground truth
  err_copy   : frame 0 repeated vs ground truth (static-world baseline)
  div_action : real_s0 vs shuf_s0 (same seed)   div_seed : real_s0 vs real_s1..
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from toolkits.libero_safety.wan_ac_sft.sampler import (
    NULL_ACTION,
    QUADRANTS,
    load_frames,
    load_manifest,
    scale_pose,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--dit", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-quadrant", type=int, default=5)
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--pose_scale", type=float, default=1.0,
                    help="must match the value the checkpoint was trained with")
    ap.add_argument("--pose_clip", type=float, default=4.0, help="must match training (0 = no cap)")
    ap.add_argument("--view", default="agent", choices=["agent", "dual"],
                    help="dual: agent stacked over wrist (dual_view.py); metrics per view")
    ap.add_argument("--dual_roots", action="store_true",
                    help="only roots with a wrist view (same root set as --view dual, for agent-view comparison)")
    args = ap.parse_args()
    dual = args.view == "dual"

    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline
    import lpips

    dev = "cuda"
    pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=dev, model_configs=[
        ModelConfig(path=args.dit, offload_device="cpu"), ModelConfig(path=args.vae, offload_device="cpu")])
    pipe.dit.to(dev)
    pipe.vae.to(dev)
    if dual:
        from toolkits.libero_safety.wan_ac_sft.dual_view import enable_dual
        enable_dual(pipe)
    metric = lpips.LPIPS(net="alex").to(dev).eval()

    T = args.chunks * 8
    rows = [r for r in load_manifest(args.index, "test") if r["kind"] == "root" and r["num_frames"] >= T + 1]
    if dual or args.dual_roots:
        rows = [r for r in rows if r.get("dual") and r.get("wrist_segments")]
    H = 512 if dual else 256
    rng = np.random.default_rng(0)
    picks = []
    for q in QUADRANTS:
        qr = [r for r in rows if r["quadrant"] == q]
        picks += [qr[i] for i in rng.choice(len(qr), size=min(args.per_quadrant, len(qr)), replace=False)]

    def actions_of(r):
        with np.load(r["labels"]) as lab:
            return scale_pose(lab["actions"][:T].astype(np.float32), args.pose_scale, args.pose_clip)

    def rollout(f0, acts, seed):
        anchor = Image.fromarray(f0)
        memory = [anchor] * 4
        mem_act = np.repeat(NULL_ACTION[None], 4, 0)
        frames = []
        for c in range(args.chunks):
            chunk_act = acts[8 * c: 8 * c + 8]
            a = np.concatenate([NULL_ACTION[None], mem_act, chunk_act], 0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = pipe(seed=seed + 1000 * c, tiled=False, input_image=[anchor], input_image4=[memory],
                           action=torch.from_numpy(a).unsqueeze(0).to(dev), height=H, width=256,
                           num_frames=13, num_inference_steps=args.steps, cfg_scale=1.0,
                           progress_bar_cmd=lambda x: x, batch_size=1)
            new = [np.asarray(im, np.uint8) for im in out[0][5:]]
            frames += new
            memory = [Image.fromarray(x) for x in new[-4:]]
            mem_act = chunk_act[-4:]
        return np.stack(frames)

    def lp_chunks(a, b):
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float().to(dev) / 127.5 - 1
        with torch.no_grad():
            d = metric(t(a), t(b)).flatten().cpu().numpy()
        return d.reshape(args.chunks, 8).mean(1)

    def view_metrics(real, shuf, f0, fut):
        return {"err_real": lp_chunks(real[0], fut).tolist(), "err_shuf": lp_chunks(shuf[0], fut).tolist(),
                "err_copy": lp_chunks(np.repeat(f0[None], T, 0), fut).tolist(),
                "div_action": lp_chunks(real[0], shuf[0]).tolist(),
                "div_seed": np.mean([lp_chunks(real[0], real[s]) for s in range(1, args.seeds)], 0).tolist(),
                "div_action_s1": lp_chunks(real[1], shuf[1]).tolist()}

    results, t0 = [], time.time()
    for k, r in enumerate(picks):
        gt = load_frames(r["agent_segments"], list(range(T + 1)), "agentview_image")
        if dual:
            gw = load_frames(r["wrist_segments"], list(range(T + 1)), "robot0_eye_in_hand_image")
            gt_in = np.concatenate([gt, gw], axis=1)       # agent over wrist
        else:
            gt_in = gt
        others = [o for o in rows if o["task"] != r["task"] or o["suite"] != r["suite"]]
        other = others[int(rng.integers(len(others)))]
        real = [rollout(gt_in[0], actions_of(r), s) for s in range(args.seeds)]
        shuf = [rollout(gt_in[0], actions_of(other), s) for s in range(2)]
        top = lambda xs: [x[:, :256] for x in xs]
        rec = {"idx": r["idx"], "cell": r["cell"], "task": r["task"], "quadrant": r["quadrant"],
               "other_idx": other["idx"]}
        rec.update(view_metrics(top(real), top(shuf), gt[0], gt[1:]))   # agent view (main keys)
        if dual:
            bot = lambda xs: [x[:, 256:] for x in xs]
            rec["wrist"] = view_metrics(bot(real), bot(shuf), gw[0], gw[1:])
        results.append(rec)
        print(f"[{k + 1}/{len(picks)}] {time.time() - t0:.0f}s {r['cell']} {r['quadrant']} "
              f"last-chunk err_real={rec['err_real'][-1]:.3f} err_copy={rec['err_copy'][-1]:.3f} "
              f"div_action={rec['div_action'][-1]:.3f} div_seed={rec['div_seed'][-1]:.3f}", flush=True)

    keys = ("err_real", "err_shuf", "err_copy", "div_action", "div_seed", "div_action_s1")

    def summarize(recs):
        curve = {k: np.mean([x[k] for x in recs], 0).tolist() for k in keys}
        ratio = (np.array(curve["div_action"]) / np.maximum(np.array(curve["div_seed"]), 1e-9)).tolist()
        return {"curve": curve, "div_action_over_div_seed": ratio,
                "frac_real_beats_shuf": [float(np.mean([x["err_real"][c] < x["err_shuf"][c] for x in recs]))
                                         for c in range(args.chunks)],
                "frac_real_beats_copy": [float(np.mean([x["err_real"][c] < x["err_copy"][c] for x in recs]))
                                         for c in range(args.chunks)]}

    summary = {"n": len(results), "chunks": args.chunks, "seeds": args.seeds, "view": args.view,
               "dual_roots": bool(dual or args.dual_roots), **summarize(results)}
    if dual:
        summary["wrist"] = summarize([x["wrist"] for x in results])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "episodes": results}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
