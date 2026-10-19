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

"""P1/P3 pre-check: image orientation and action alignment for a released Wan WM.

On held-out windows (anchor frame 0 + 4 memory frames + 8 future frames), the
checkpoint predicts one 8-frame chunk. LPIPS of the prediction against the real
future is compared across:
  orient  : frames as stored (policy view) vs rotated 180 degrees
  shift   : action[f] = audit_action[f - 1] (sampler convention) vs audit_action[f]
plus a copy-last-memory-frame baseline. Lower LPIPS identifies the convention the
checkpoint was trained with.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from toolkits.libero_safety.wan_ac_sft.sampler import NULL_ACTION, load_frames, load_manifest


def actions_for(actions: np.ndarray, frame_ids: list[int], shift: int) -> np.ndarray:
    out = np.empty((len(frame_ids), actions.shape[1]), np.float32)
    for i, f in enumerate(frame_ids):
        j = f - shift
        out[i] = NULL_ACTION if i == 0 or not 0 <= j < len(actions) else actions[j]
    return out


def to_tensor(frames: np.ndarray) -> torch.Tensor:
    """uint8 [T, H, W, 3] -> float [-1, 1] [T, 3, H, W]."""
    return torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--ckpt", required=True, help="dir with model-00001.safetensors and Wan2.2_VAE.pth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()

    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline
    import lpips

    device = "cuda"
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=device,
        model_configs=[ModelConfig(path=f"{args.ckpt}/model-00001.safetensors", offload_device="cpu"),
                       ModelConfig(path=f"{args.ckpt}/Wan2.2_VAE.pth", offload_device="cpu")])
    pipe.dit.to(device)
    pipe.vae.to(device)
    metric = lpips.LPIPS(net="alex").to(device).eval()

    rows = [r for r in load_manifest(args.index, "test") if r["num_frames"] >= 60]
    rng = np.random.default_rng(args.seed)
    picks = [rows[i] for i in rng.choice(len(rows), size=min(args.n, len(rows)), replace=False)]
    variants = [("asis", 1), ("asis", 0), ("rot180", 1), ("rot180", 0)]
    results, t0 = [], time.time()
    for k, row in enumerate(picks):
        s = int(rng.integers(1, row["num_frames"] - 12))
        ids = [0] + list(range(s, s + 12))
        frames = load_frames(row["agent_segments"], ids, "agentview_image")
        with np.load(row["labels"]) as lab:
            audit_actions = lab["actions"]
        rec = {"idx": row["idx"], "cell": row["cell"], "task": row["task"], "quadrant": row["quadrant"], "start": s}
        for orient, shift in variants:
            fr = frames if orient == "asis" else np.ascontiguousarray(frames[:, ::-1, ::-1])
            imgs = [Image.fromarray(f) for f in fr]
            act = torch.from_numpy(actions_for(audit_actions, ids, shift)).unsqueeze(0).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = pipe(seed=args.seed, tiled=False, input_image=[imgs[0]], input_image4=[imgs[1:5]],
                           action=act, height=256, width=256, num_frames=13,
                           num_inference_steps=args.steps, cfg_scale=1.0,
                           progress_bar_cmd=lambda x: x, batch_size=1)
            pred = np.stack([np.asarray(im, np.uint8) for im in out[0][5:]])
            with torch.no_grad():
                d = metric(to_tensor(pred).to(device), to_tensor(fr[5:]).to(device)).flatten().cpu().numpy()
            rec[f"{orient}_shift{shift}"] = float(d.mean())
        with torch.no_grad():
            copy = np.repeat(frames[4:5], 8, axis=0)
            rec["copy_last"] = float(metric(to_tensor(copy).to(device), to_tensor(frames[5:]).to(device)).mean())
        results.append(rec)
        print(f"[{k + 1}/{len(picks)}] {time.time() - t0:.0f}s " +
              " ".join(f"{v}={rec[v]:.3f}" for v in rec if v.startswith(("asis", "rot", "copy"))), flush=True)

    keys = [f"{o}_shift{s}" for o, s in variants] + ["copy_last"]
    summary = {k: {"mean": float(np.mean([r[k] for r in results])),
                   "median": float(np.median([r[k] for r in results]))} for k in keys}
    wins = {k: int(sum(min(keys[:4], key=lambda x: r[x]) == k for r in results)) for k in keys[:4]}
    report = {"n": len(results), "steps": args.steps, "summary": summary, "per_window_best": wins, "windows": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({"summary": summary, "per_window_best": wins}, indent=1))


if __name__ == "__main__":
    main()
