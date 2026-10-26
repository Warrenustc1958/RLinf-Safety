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

"""SC-GRPO proprio state from frames: CNN regressor dual frame -> pi0.5 state (8).

Trained on real frames of Wan-seen train groups (cost_train / h_train / dev), targets
normalized with the pi0.5 norm stats. Evaluated on
  real      held-out real frames
  gen_pilot decoded generated futures of pairs/pilot_v1 (Wan-seen init states), TF and AR
  gen_fresh decoded generated futures of pairs/h_deep_fresh4 (Wan-unseen), AR depth <= 32
against the true state at the same frame, with the "hold the keyframe state" baseline.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from toolkits.libero_safety.sam_alignment.cache_real import VAE, load_dual_frames, load_vae
from toolkits.libero_safety.sam_alignment.schema import OUT_ROOT
from toolkits.libero_safety.sam_alignment.scgrpo.build_keyframes import ROLES, episode_states

NORM = "/mnt/nfs/fjj/Code/rlinf-safety-runtime/checkpoints/pi05_libero_safety_rlinf/assets/lerobot/norm_stats.json"


def down(fr):
    """[..., 512, 256, 3] uint8 -> [..., 256, 128, 3] uint8 (2x2 mean)."""
    s = fr.shape
    x = fr.reshape(*s[:-3], s[-3] // 2, 2, s[-2] // 2, 2, 3).astype(np.float32).mean((-4, -2))
    return x.astype(np.uint8)


def sample_ep(job):
    e, k, seed = job
    rng = np.random.default_rng(seed)
    st = episode_states(e)
    fr = load_dual_frames(e)
    idx = np.sort(rng.choice(len(st), min(k, len(st)), replace=False))
    return down(fr[idx]), st[idx]


class StateNet(nn.Module):
    def __init__(self):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.GELU(),
                                 nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o), nn.GELU())
        self.f = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128), blk(128, 256), blk(256, 256),
                               nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.h = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 8))

    def forward(self, x):  # x uint8 [B, 256, 128, 3]
        x = x.permute(0, 3, 1, 2).float().div(127.5).sub(1)
        return self.h(self.f(x))


def predict(net, frames, bs=512):
    out = []
    with torch.no_grad():
        for i in range(0, len(frames), bs):
            out.append(net(torch.from_numpy(np.ascontiguousarray(frames[i:i + bs])).cuda()).float().cpu().numpy())
    return np.concatenate(out)


def decode_rows(pipe, z, rows, device, bs=8):
    """Generated latents [N,48,4,32,16] -> decoded future frames [N, 8, 256, 128, 3] (downsampled)."""
    out = []
    for i in range(0, len(rows), bs):
        lat = torch.from_numpy(np.asarray(z[np.sort(rows[i:i + bs])], np.float32)).to(device, torch.bfloat16)
        with torch.no_grad():
            v = pipe.vae.decode(lat, device=device, tiled=False)  # [B, 3, 13, 512, 256] in [-1, 1]
        v = ((v.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 4, 1).cpu().numpy()
        out.append(down(v[:, 5:13]))
    order = np.argsort(np.argsort(rows))
    return np.concatenate(out)[order] if out else np.zeros((0, 8, 256, 128, 3), np.uint8)


def gen_eval(net, pipe, pdir, labels, mu, sd, device, n_tf, n_ar, max_depth, rng):
    pm = dict(np.load(Path(pdir) / "meta.npz"))
    z = np.load(Path(pdir) / "gen_z.npy", mmap_mode="r")
    eps = {json.loads(l)["uid"]: json.loads(l) for l in open(labels)}
    res = {}
    for name, m, n in (("tf", pm["mode"] == 0, n_tf), ("ar", (pm["mode"] == 1) & (pm["depth"] <= max_depth), n_ar)):
        cand = np.flatnonzero(m)
        rows = np.sort(rng.choice(cand, min(n, len(cand)), replace=False))
        fr = decode_rows(pipe, z, rows, device)
        errs, hold, depth = [], [], []
        st_cache = {}
        for r, f8 in zip(rows, fr):
            u = str(pm["uid"][r])
            if u not in st_cache:
                st_cache[u] = episode_states(eps[u])
            st = st_cache[u]
            t0 = int(pm["t0"][r])
            fut = np.clip(np.arange(t0 + 1, t0 + 9), 0, len(st) - 1)
            p = predict(net, f8) * sd + mu
            errs.append(np.abs(p - st[fut]))
            ref = st[0] if name == "ar" else st[t0]  # AR imagines from frame 0, TF from t0
            hold.append(np.abs(ref[None] - st[fut]))
            depth.append(int(pm["depth"][r]))
        errs, hold, depth = np.stack(errs), np.stack(hold), np.array(depth)
        r = {"rows": int(len(rows)), "pos_mae": float(errs[..., :3].mean()), "rot_mae": float(errs[..., 3:6].mean()),
             "grip_mae": float(errs[..., 6:8].mean()), "hold_pos_mae": float(hold[..., :3].mean()),
             "hold_rot_mae": float(hold[..., 3:6].mean())}
        if name == "ar":
            for lo, hi in ((1, 4), (5, 8), (9, 16), (17, 32)):
                s = (depth >= lo) & (depth <= hi)
                if s.any():
                    r[f"d{lo}-{hi}"] = {"n": int(s.sum()), "pos_mae": float(errs[s][..., :3].mean()),
                                        "hold_pos_mae": float(hold[s][..., :3].mean()),
                                        "grip_mae": float(errs[s][..., 6:8].mean())}
        res[name] = r
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT_ROOT / "scgrpo" / "state_reg_v1"))
    ap.add_argument("--n_eps", type=int, default=2500)
    ap.add_argument("--per_ep", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    norm = json.load(open(NORM))
    norm = norm.get("norm_stats", norm)["state"]
    mu, sd = np.array(norm["mean"], np.float32), np.array(norm["std"], np.float32)
    reg = json.load(open(OUT_ROOT / "splits" / "split_registry.json"))
    eps = [json.loads(l) for l in open(OUT_ROOT / "labels" / "episodes.jsonl")]
    eps = [e for e in eps if reg["episode_role"].get(e["uid"]) in ROLES and e["label_source"] == "grader_v1"
           and e["dual"]]
    rng = np.random.default_rng(0)
    eps = [eps[i] for i in rng.permutation(len(eps))[: args.n_eps]]
    groups = sorted({e["group"] for e in eps})
    val_g = set(rng.choice(groups, len(groups) // 10, replace=False))
    with Pool(args.workers) as p:
        data = p.map(sample_ep, [(e, args.per_ep, i) for i, e in enumerate(eps)], chunksize=4)
    is_val = np.concatenate([[e["group"] in val_g] * len(d[1]) for e, d in zip(eps, data)])
    X = np.concatenate([d[0] for d in data])
    Y = (np.concatenate([d[1] for d in data]) - mu) / sd
    Xtr, Ytr, Xva, Yva = X[~is_val], Y[~is_val], X[is_val], Y[is_val]
    print(f"frames train {len(Xtr)} val {len(Xva)}", flush=True)

    net = StateNet().cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = args.epochs * (len(Xtr) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
    Ytr_t = torch.from_numpy(Ytr.astype(np.float32))
    for ep in range(args.epochs):
        net.train()
        perm = np.random.permutation(len(Xtr))
        for i in range(0, len(perm) - 255, 256):
            j = perm[i:i + 256]
            x = torch.from_numpy(Xtr[j]).cuda()
            if np.random.rand() < 0.5:  # mild photometric jitter: generated frames are softer
                x = (x.float() * float(np.random.uniform(0.85, 1.15))).clamp(0, 255).byte()
            loss = nn.functional.smooth_l1_loss(net(x), Ytr_t[j].cuda())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        pv = predict(net, Xva) * sd + mu
        tv = Yva * sd + mu
        print(f"ep {ep} val pos_mae {np.abs(pv - tv)[:, :3].mean():.4f} rot_mae {np.abs(pv - tv)[:, 3:6].mean():.4f} "
              f"grip_mae {np.abs(pv - tv)[:, 6:8].mean():.5f}", flush=True)
    torch.save({"model": net.state_dict(), "mu": mu, "sd": sd}, out / "state_reg.pt")

    pv = predict(net, Xva) * sd + mu
    tv = Yva * sd + mu
    rep = {"real_val": {"frames": int(len(Xva)), "pos_mae": float(np.abs(pv - tv)[:, :3].mean()),
                        "rot_mae": float(np.abs(pv - tv)[:, 3:6].mean()),
                        "grip_mae": float(np.abs(pv - tv)[:, 6:8].mean())}}
    device = "cuda"
    pipe = load_vae(VAE, device)
    rep["gen_pilot"] = gen_eval(net, pipe, OUT_ROOT / "pairs" / "pilot_v1", OUT_ROOT / "labels" / "episodes.jsonl",
                                mu, sd, device, 300, 300, 16, rng)
    rep["gen_fresh_hdeep"] = gen_eval(net, pipe, OUT_ROOT / "pairs" / "h_deep_fresh4",
                                      OUT_ROOT / "fresh4" / "labels" / "episodes.jsonl", mu, sd, device, 300, 500, 32, rng)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
