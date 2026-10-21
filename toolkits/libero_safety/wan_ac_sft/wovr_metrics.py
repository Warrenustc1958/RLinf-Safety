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

"""WoVR-style video-world-model metrics for Wan AC-SFT checkpoints.

For each rollout horizon ``H`` the checkpoint is run autoregressively from
held-out test roots with the episode's own actions (chunk 8, 4 memory frames,
anchor = root frame 0, 5 denoising steps -- the same regime as the WM backend),
and the generated frames are compared with the ground truth of the same roots:

  LPIPS     per-frame LPIPS(pred_t, gt_t), AlexNet backbone
  FID       Frechet distance of Inception-V3 pool3 features (299x299, per frame)
  FVD       Frechet distance of I3D features (32-frame clips at 224x224)
  FloLPIPS  LPIPS between RAFT optical-flow images of pred and gt, the flow
            rendered as a 3-channel image scaled by ``--flow-scale``
  FPS       generated frames per second of generation wall clock, plus the
            real-time factor at 20 Hz control

Reference distributions are the ground truth of the *same* roots, so FID/FVD
read as "distance to this held-out set", not to the whole LIBERO corpus. Three
sequences are scored per root: ``pred`` (the model), ``copy`` (root frame 0
repeated -- the static baseline) and ``gt`` (the reference distribution).

``FloLPIPS`` follows the paper's definition: optical flow is estimated on the
model sequence and on the real sequence independently, and the flow fields are
compared with LPIPS. A second, complementary number ``warp_lpips`` (LPIPS
between the gt-flow-warped prediction and the next gt frame) is reported too,
because it answers the question "does the prediction move the way reality does"
rather than "does the prediction move consistently with itself".
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

from toolkits.libero_safety.wan_ac_sft.sampler import (
    NULL_ACTION,
    QUADRANTS,
    load_frames,
    load_manifest,
    scale_pose,
)

CONTROL_HZ = 20.0  # LIBERO runs at 20 Hz, used for the real-time factor


# --------------------------------------------------------------------------- #
# feature extractors
# --------------------------------------------------------------------------- #
class InceptionFeatures:
    """Inception-V3 pool3 (2048-d) for FID, as in the standard torchmetrics setup."""

    def __init__(self, device, torch_home):
        import os

        from torchvision.models import Inception_V3_Weights, inception_v3

        os.environ.setdefault("TORCH_HOME", torch_home)
        self.net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        self.net.fc = torch.nn.Identity()
        self.net.eval().to(device)
        self.device = device
        self._feat = None
        self.net.avgpool.register_forward_hook(self._hook)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _hook(self, _m, inp, _out):
        self._feat = Fn.adaptive_avg_pool2d(inp[0], 1).flatten(1)

    @torch.no_grad()
    def __call__(self, frames_u8):
        """frames_u8: [N, H, W, 3] uint8 numpy."""
        x = torch.from_numpy(np.ascontiguousarray(frames_u8)).permute(0, 3, 1, 2).to(self.device)
        x = x.float() / 255.0
        if x.shape[-1] != 299:
            x = Fn.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        self.net(x)
        return self._feat.float()


class I3DFeatures:
    """I3D (Kinetics-400, torchscript, logits head) for FVD."""

    def __init__(self, device, path):
        self.net = torch.jit.load(path, map_location=device).eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, clip_u8):
        """clip_u8: [B, T, H, W, 3] uint8 numpy, T >= 16."""
        x = torch.from_numpy(np.ascontiguousarray(clip_u8)).permute(0, 4, 1, 2, 3).to(self.device)
        x = x.float() / 255.0
        if x.shape[-1] != 224:
            b, c, t, h, w = x.shape
            x = Fn.interpolate(x.reshape(b * t, c, h, w), size=(224, 224), mode="bilinear",
                               align_corners=False).reshape(b, c, t, 224, 224)
        return self.net(x).float()


class RaftFlow:
    """RAFT optical flow, returns the final flow at input resolution."""

    def __init__(self, device):
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        self.net = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2, progress=False).eval().to(device)
        self.device = device
        self.weights = Raft_Large_Weights.C_T_SKHT_V2

    @torch.no_grad()
    def __call__(self, a_u8, b_u8):
        """a_u8, b_u8: [N, H, W, 3] uint8 -> flow [N, 2, H, W] float."""
        prep = self.weights.transforms()
        a = torch.from_numpy(np.ascontiguousarray(a_u8)).permute(0, 3, 1, 2).to(self.device)
        b = torch.from_numpy(np.ascontiguousarray(b_u8)).permute(0, 3, 1, 2).to(self.device)
        a = a.float() / 255.0
        b = b.float() / 255.0
        # raft wants even, /8 sizes
        h, w = a.shape[-2] // 8 * 8, a.shape[-1] // 8 * 8
        if (h, w) != tuple(a.shape[-2:]):
            a = Fn.interpolate(a, size=(h, w), mode="bilinear", align_corners=False)
            b = Fn.interpolate(b, size=(h, w), mode="bilinear", align_corners=False)
        a, b = prep(a, b)
        return self.net(a, b)[-1]


# --------------------------------------------------------------------------- #
# frechet distance
# --------------------------------------------------------------------------- #
def frechet(feats_a: np.ndarray, feats_b: np.ndarray) -> float:
    from scipy.linalg import sqrtm

    a, b = feats_a.astype(np.float64), feats_b.astype(np.float64)
    mu1, mu2 = a.mean(0), b.mean(0)
    c1, c2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean, _ = sqrtm(c1 @ c2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    d = mu1 - mu2
    val = float(d @ d + np.trace(c1) + np.trace(c2) - 2.0 * np.trace(covmean))
    return max(val, 0.0)


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def build_pipe(dit, vae, device):
    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

    pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=[
        ModelConfig(path=dit, offload_device="cpu"), ModelConfig(path=vae, offload_device="cpu")])
    pipe.dit.to(device)
    pipe.vae.to(device)
    return pipe


def rollout(pipe, device, f0, acts, chunks, steps, seed):
    """Autoregressive rollout, identical to the WM backend's regime."""
    anchor = Image.fromarray(f0)
    memory = [anchor] * 4
    mem_act = np.repeat(NULL_ACTION[None], 4, 0)
    frames = []
    for c in range(chunks):
        chunk_act = acts[8 * c: 8 * c + 8]
        a = np.concatenate([NULL_ACTION[None], mem_act, chunk_act], 0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = pipe(seed=seed + 1000 * c, tiled=False, input_image=[anchor], input_image4=[memory],
                       action=torch.from_numpy(a).unsqueeze(0).to(device), height=256, width=256,
                       num_frames=13, num_inference_steps=steps, cfg_scale=1.0,
                       progress_bar_cmd=lambda x: x, batch_size=1)
        new = [np.asarray(im, np.uint8) for im in out[0][5:]]
        frames += new
        memory = [Image.fromarray(x) for x in new[-4:]]
        mem_act = chunk_act[-4:]
    return np.stack(frames)


# --------------------------------------------------------------------------- #
# metric accumulation
# --------------------------------------------------------------------------- #
def sequence_metrics(pred, gt, anchor, feats):
    """Return the per-frame metric values for one rollout."""
    import lpips

    dev = feats["device"]
    H = len(gt)

    def to_t(x):
        return torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float().to(dev) / 127.5 - 1.0

    def lpips_pairs(a, b):
        with torch.no_grad():
            return feats["lpips"](to_t(a), to_t(b)).flatten().cpu().numpy()

    out = {}

    # LPIPS per generated frame
    out["lpips"] = lpips_pairs(pred, gt)
    out["lpips_copy"] = lpips_pairs(np.repeat(anchor[None], H, 0), gt)

    # FID features (per frame)
    out["fid_pred"] = feats["inception"](pred).cpu().numpy()
    out["fid_copy"] = feats["inception"](np.repeat(anchor[None], H, 0)).cpu().numpy()
    out["fid_gt"] = feats["inception"](gt).cpu().numpy()

    # FVD features (16-frame clips, no overlap -- the I3D / torchmetrics default)
    clip = feats["fvd_clip"]
    n_clip = H // clip
    fvd_pred, fvd_copy, fvd_gt = [], [], []
    for c in range(n_clip):
        sl = slice(c * clip, (c + 1) * clip)
        fvd_pred.append(feats["i3d"](pred[sl][None]))
        fvd_gt.append(feats["i3d"](gt[sl][None]))
    if n_clip:
        out["fvd_pred"] = torch.cat(fvd_pred).cpu().numpy()
        out["fvd_gt"] = torch.cat(fvd_gt).cpu().numpy()

    # Flow-based metrics
    a_prev, a_next = pred[:-1], pred[1:]
    g_prev, g_next = gt[:-1], gt[1:]
    flo_pred, flo_gt, warp_lp = [], [], []
    bs = 4
    for i in range(0, len(a_prev), bs):
        fp = feats["raft"](a_prev[i:i + bs], a_next[i:i + bs])
        fg = feats["raft"](g_prev[i:i + bs], g_next[i:i + bs])
        # flow rendered as an image, both ends scaled identically
        s = feats["flow_scale"]
        ip = torch.cat([torch.clamp(fp / s, -1, 1), torch.zeros_like(fp[:, :1])], 1)
        ig = torch.cat([torch.clamp(fg / s, -1, 1), torch.zeros_like(fg[:, :1])], 1)
        with torch.no_grad():
            flo_pred.append(feats["lpips"](ip, ig).flatten().cpu().numpy())
        # warp the prediction with the *gt* flow, compare against the next gt frame
        yy, xx = torch.meshgrid(torch.arange(fp.shape[2], device=dev),
                                torch.arange(fp.shape[3], device=dev), indexing="ij")
        grid = torch.stack([(xx + fg[:, 0]) / (fp.shape[3] - 1) * 2 - 1,
                            (yy + fg[:, 1]) / (fp.shape[2] - 1) * 2 - 1], -1)
        warped = Fn.grid_sample(to_t(a_prev[i:i + bs]), grid, align_corners=True)
        with torch.no_grad():
            warp_lp.append(feats["lpips"](warped, to_t(g_next[i:i + bs])).flatten().cpu().numpy())
    out["flo_lpips"] = np.concatenate(flo_pred)
    out["warp_lpips"] = np.concatenate(warp_lp)
    return out


def pick_roots(rows, horizon, per_quadrant, seed):
    rng = np.random.default_rng(seed)
    cand = [r for r in rows if r["kind"] == "root" and r["num_frames"] >= horizon + 1]
    picks = []
    for q in QUADRANTS:
        qr = [r for r in cand if r["quadrant"] == q]
        if not qr:
            continue
        n = min(per_quadrant, len(qr))
        picks += [qr[int(i)] for i in rng.choice(len(qr), size=n, replace=False)]
    return picks


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", required=True)
    ap.add_argument("--dit", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizons", default="128,256", help="comma-separated rollout lengths in frames")
    ap.add_argument("--per-quadrant", type=int, default=8)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--pose_scale", type=float, default=1.0,
                    help="must match the value the checkpoint was trained with")
    ap.add_argument("--pose_clip", type=float, default=4.0, help="must match training (0 = no cap)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--metric-ckpt", default="/mnt/nfs/fjj/wan_ac_sft/metric_ckpt")
    ap.add_argument("--flow-scale", type=float, default=20.0, help="pixels mapped to +-1 in the flow image")
    ap.add_argument("--fvd-clip", type=int, default=16, help="frames per FVD clip (I3D needs >= 16)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import lpips

    dev = args.device
    feats = {
        "device": dev,
        "lpips": lpips.LPIPS(net="alex").to(dev).eval(),
        "inception": InceptionFeatures(dev, str(Path(args.metric_ckpt) / "torch")),
        "i3d": I3DFeatures(dev, str(Path(args.metric_ckpt) / "i3d_torchscript.pt")),
        "raft": RaftFlow(dev),
        "flow_scale": args.flow_scale,
        "fvd_clip": args.fvd_clip,
    }
    pipe = build_pipe(args.dit, args.vae, dev)

    rows = load_manifest(args.index, "test")
    report = {"dit": args.dit, "tag": args.tag, "steps": args.steps, "horizons": {}}

    for H in [int(x) for x in args.horizons.split(",")]:
        chunks = H // 8
        picks = pick_roots(rows, H, args.per_quadrant, args.seed)
        if not picks:
            print(f"[H={H}] no test root is long enough, skipped", flush=True)
            continue
        byq = defaultdict(int)
        for r in picks:
            byq[r["quadrant"]] += 1

        acc = defaultdict(list)
        t_gen = 0.0
        t0 = time.time()
        for k, r in enumerate(picks):
            gt_all = load_frames(r["agent_segments"], list(range(H + 1)), "agentview_image")
            f0, gt = gt_all[0], gt_all[1:]
            with np.load(r["labels"]) as lab:
                acts = scale_pose(lab["actions"][:H].astype(np.float32), args.pose_scale, args.pose_clip)
            if len(acts) < H:
                acts = np.concatenate([acts, np.zeros((H - len(acts), 7), np.float32)], 0)

            g0 = time.time()
            pred = rollout(pipe, dev, f0, acts, chunks, args.steps, args.seed)
            t_gen += time.time() - g0

            m = sequence_metrics(pred, gt, f0, feats)
            for key, val in m.items():
                acc[key].append(val)
            print(f"[H={H}] {k + 1}/{len(picks)} {r['cell']} {r['quadrant']} "
                  f"lpips={m['lpips'].mean():.4f} copy={m['lpips_copy'].mean():.4f} "
                  f"flo={m['flo_lpips'].mean():.4f} gen={time.time() - g0:.0f}s", flush=True)

        res = {
            "n_roots": len(picks),
            "by_quadrant": dict(byq),
            "lpips": float(np.concatenate(acc["lpips"]).mean()),
            "lpips_copy": float(np.concatenate(acc["lpips_copy"]).mean()),
            "flo_lpips": float(np.concatenate(acc["flo_lpips"]).mean()),
            "warp_lpips": float(np.concatenate(acc["warp_lpips"]).mean()),
            "fid": frechet(np.concatenate(acc["fid_pred"]), np.concatenate(acc["fid_gt"])),
            "fid_copy": frechet(np.concatenate(acc["fid_copy"]), np.concatenate(acc["fid_gt"])),
            "fps": H * len(picks) / max(t_gen, 1e-9),
            "realtime_factor_at_20hz": (H / CONTROL_HZ) / max(t_gen, 1e-9),
            "n_fid_samples": int(sum(len(x) for x in acc["fid_pred"])),
        }
        if acc["fvd_pred"]:
            res["fvd"] = frechet(np.concatenate(acc["fvd_pred"]), np.concatenate(acc["fvd_gt"]))
            res["n_fvd_clips"] = int(sum(len(x) for x in acc["fvd_pred"]))
        res["wall_s"] = round(time.time() - t0, 1)
        report["horizons"][str(H)] = res
        print(json.dumps({str(H): res}, indent=1), flush=True)

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1))

    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
