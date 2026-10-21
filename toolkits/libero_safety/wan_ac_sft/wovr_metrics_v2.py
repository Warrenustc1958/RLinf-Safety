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

"""WoVR-protocol video metrics, v2: standard FID / FVD and FloLPIPS.

Differences from wovr_metrics.py (v1), whose FID / FVD / FloLPIPS are not on the
published scale:
  FID       pytorch-fid InceptionV3 (pt_inception-2015-12-05) pool3, inputs in [0, 1]
            (v1 used torchvision's ImageNet Inception);
  FVD       I3D torchscript features (return_features=True, 400-d) on 16-frame clips
            in [-1, 1] at 224x224, as in StyleGAN-V / common_metrics_on_video_quality
            (v1 used the softmax output on [0, 1] inputs, which shrinks FVD ~1000x);
  FloLPIPS  Danier et al. 2022: the spatial LPIPS map between prediction and ground
            truth, averaged with weights |flow(gt[t-1] -> gt[t]) - flow(pred[t-1] -> pred[t])|.
            Reimplemented with RAFT instead of PWC-Net and map-level weighting, so
            values are close to, not bit-identical with, the reference code.
LPIPS (AlexNet, per frame) and the rollout protocol (chunk 8, 4 memory frames, anchor
= frame 0, 5 denoising steps) are unchanged. FID / FVD use a few thousand frames / a
few hundred clips here and are biased upward at that sample size, so only compare
checkpoints evaluated with the same --per-quadrant.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from toolkits.libero_safety.wan_ac_sft.sampler import QUADRANTS, load_frames, load_manifest, scale_pose
from toolkits.libero_safety.wan_ac_sft.wovr_metrics import RaftFlow, build_pipe, frechet, pick_roots, rollout

CONTROL_HZ = 20.0


def to_unit(x_u8, device):
    """[N, H, W, 3] uint8 -> [N, 3, H, W] float in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(x_u8)).permute(0, 3, 1, 2).to(device).float() / 255.0


class Metrics:
    def __init__(self, device, metric_ckpt, fid_lib, fvd_clip):
        os.environ["TORCH_HOME"] = str(Path(metric_ckpt) / "torch")
        if fid_lib and fid_lib not in sys.path:
            sys.path.insert(0, fid_lib)
        import lpips
        from pytorch_fid.inception import InceptionV3

        self.dev, self.fvd_clip = device, fvd_clip
        self.lpips = lpips.LPIPS(net="alex").to(device).eval()
        self.lpips_map = lpips.LPIPS(net="alex", spatial=True).to(device).eval()
        self.inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
        self.i3d = torch.jit.load(str(Path(metric_ckpt) / "i3d_torchscript.pt"), map_location=device).eval()
        self.raft = RaftFlow(device)

    @torch.no_grad()
    def lpips_frames(self, a, b):
        out = []
        for i in range(0, len(a), 32):
            out.append(self.lpips(to_unit(a[i:i + 32], self.dev) * 2 - 1,
                                  to_unit(b[i:i + 32], self.dev) * 2 - 1).flatten())
        return torch.cat(out).cpu().numpy()

    @torch.no_grad()
    def fid_feats(self, x):
        out = []
        for i in range(0, len(x), 64):
            out.append(self.inception(to_unit(x[i:i + 64], self.dev))[0].flatten(1))
        return torch.cat(out).cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def fvd_feats(self, x):
        c, n = self.fvd_clip, len(x) // self.fvd_clip
        if n == 0:
            return np.zeros((0, 400))
        out = []
        for k in range(n):
            v = to_unit(x[k * c:(k + 1) * c], self.dev)                  # [T, 3, H, W]
            v = Fn.interpolate(v, size=(224, 224), mode="bilinear", align_corners=False)
            v = (v * 2 - 1).permute(1, 0, 2, 3).unsqueeze(0)             # [1, 3, T, 224, 224]
            out.append(self.i3d(v, rescale=False, resize=False, return_features=True))
        return torch.cat(out).cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def flolpips(self, pred, gt, anchor):
        """Each video's own motion: flow(gt[t-1] -> gt[t]) vs flow(pred[t-1] -> pred[t])."""
        prev_gt = np.concatenate([anchor[None], gt[:-1]], 0)
        prev_pred = np.concatenate([anchor[None], pred[:-1]], 0)
        out = []
        for i in range(0, len(gt), 8):
            g, q = gt[i:i + 8], pred[i:i + 8]
            f_ref = self.raft(prev_gt[i:i + 8], g)
            f_dis = self.raft(prev_pred[i:i + 8], q)
            w = (f_ref - f_dis).norm(dim=1, keepdim=True)                # [N, 1, h, w]
            lmap = self.lpips_map(to_unit(q, self.dev) * 2 - 1, to_unit(g, self.dev) * 2 - 1)
            if w.shape[-2:] != lmap.shape[-2:]:
                w = Fn.interpolate(w, size=lmap.shape[-2:], mode="bilinear", align_corners=False)
            out.append(((lmap * w).sum((1, 2, 3)) / (w.sum((1, 2, 3)) + 1e-8)))
        return torch.cat(out).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", required=True)
    ap.add_argument("--dit", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizons", default="128,256")
    ap.add_argument("--per-quadrant", type=int, default=16)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pose_scale", type=float, default=1.0, help="must match the checkpoint's training value")
    ap.add_argument("--pose_clip", type=float, default=0.0, help="must match training (0 = no cap)")
    ap.add_argument("--metric-ckpt", default="/mnt/nfs/fjj/wan_ac_sft/metric_ckpt")
    ap.add_argument("--fid-lib", default="/mnt/nfs/fjj/wan_ac_sft/metric_ckpt/pylib/pytorch_fid-0.3.0-py3-none-any.whl")
    ap.add_argument("--fvd-clip", type=int, default=16)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    dev = "cuda"
    met = Metrics(dev, args.metric_ckpt, args.fid_lib, args.fvd_clip)
    pipe = build_pipe(args.dit, args.vae, dev)
    rows = load_manifest(args.index, "test")
    report = {"dit": args.dit, "tag": args.tag, "metrics_version": 2, "steps": args.steps,
              "pose_scale": args.pose_scale, "pose_clip": args.pose_clip, "horizons": {}}

    for H in [int(x) for x in args.horizons.split(",")]:
        picks = pick_roots(rows, H, args.per_quadrant, args.seed)
        if not picks:
            print(f"[H={H}] no test root is long enough, skipped", flush=True)
            continue
        acc, per_q = defaultdict(list), defaultdict(lambda: defaultdict(list))
        t_gen, t0 = 0.0, time.time()
        for k, r in enumerate(picks):
            gt_all = load_frames(r["agent_segments"], list(range(H + 1)), "agentview_image")
            f0, gt = gt_all[0], gt_all[1:]
            with np.load(r["labels"]) as lab:
                acts = lab["actions"][:H].astype(np.float32)
            if len(acts) < H:
                acts = np.concatenate([acts, np.zeros((H - len(acts), 7), np.float32)], 0)
            acts = scale_pose(acts, args.pose_scale, args.pose_clip)
            g0 = time.time()
            pred = rollout(pipe, dev, f0, acts, H // 8, args.steps, args.seed)
            t_gen += time.time() - g0
            copy = np.repeat(f0[None], H, 0)
            lp, lpc, flo = met.lpips_frames(pred, gt), met.lpips_frames(copy, gt), met.flolpips(pred, gt, f0)
            acc["lpips"].append(lp); acc["lpips_copy"].append(lpc); acc["flolpips"].append(flo)
            acc["fid_pred"].append(met.fid_feats(pred)); acc["fid_gt"].append(met.fid_feats(gt))
            acc["fvd_pred"].append(met.fvd_feats(pred)); acc["fvd_gt"].append(met.fvd_feats(gt))
            if k == 0:
                fid_copy, fvd_copy = [], []
            fid_copy.append(met.fid_feats(copy)); fvd_copy.append(met.fvd_feats(copy))
            per_q[r["quadrant"]]["lpips"].append(lp.mean()); per_q[r["quadrant"]]["flolpips"].append(flo.mean())
            print(f"[H={H}] {k + 1}/{len(picks)} {r['cell']} {r['quadrant']} lpips={lp.mean():.4f} "
                  f"copy={lpc.mean():.4f} flolpips={flo.mean():.4f} gen={time.time() - g0:.0f}s", flush=True)
        cat = lambda key: np.concatenate(acc[key])
        res = {
            "n_roots": len(picks),
            "by_quadrant": {q: len(per_q[q]["lpips"]) for q in QUADRANTS if per_q[q]["lpips"]},
            "lpips": float(cat("lpips").mean()), "lpips_copy": float(cat("lpips_copy").mean()),
            "flolpips": float(cat("flolpips").mean()),
            "fid": frechet(cat("fid_pred"), cat("fid_gt")), "fid_copy": frechet(np.concatenate(fid_copy), cat("fid_gt")),
            "fvd": frechet(cat("fvd_pred"), cat("fvd_gt")), "fvd_copy": frechet(np.concatenate(fvd_copy), cat("fvd_gt")),
            "n_fid_frames": int(len(cat("fid_pred"))), "n_fvd_clips": int(len(cat("fvd_pred"))),
            "fps": H * len(picks) / max(t_gen, 1e-9),
            "realtime_factor_at_20hz": (H / CONTROL_HZ) * len(picks) / max(t_gen, 1e-9),
            "per_quadrant": {q: {m: float(np.mean(v)) for m, v in d.items()} for q, d in per_q.items()},
            "wall_s": round(time.time() - t0, 1),
        }
        report["horizons"][str(H)] = res
        print(json.dumps({str(H): res}, indent=1), flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
