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

"""S1 evaluation of one Wan checkpoint on held-out (test) windows.

Per window (anchor + 4 memory + 8 future frames, quadrant-stratified):
  real   : LPIPS(pred with real actions, real future)        -> fidelity
  copy   : LPIPS(last memory frame repeated, real future)     -> baseline
  shift0 : LPIPS(pred with actions not shifted, real future)  -> action alignment
  ctrl   : LPIPS(pred real actions, pred actions of another window), same seed
  noise  : LPIPS(pred real actions seed 0, pred real actions seed 1)
Controllability ratio = mean(ctrl) / mean(noise) (gate >= 2).
With --init, also reports the relative L2 change of DiT weights vs the init file.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from toolkits.libero_safety.wan_ac_sft.sampler import NULL_ACTION, QUADRANTS, load_frames, load_manifest


def acts(actions, ids, shift):
    out = np.empty((len(ids), actions.shape[1]), np.float32)
    for i, f in enumerate(ids):
        j = f - shift
        out[i] = NULL_ACTION if i == 0 or not 0 <= j < len(actions) else actions[j]
    return out


def to_t(x):
    return torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float() / 127.5 - 1.0


def weight_change(ckpt, init):
    """Relative L2 change of shared tensors; ``init`` may be comma-separated shards."""
    from safetensors import safe_open
    num = den = 0.0
    keys, init_keys = 0, set()
    with safe_open(ckpt, "pt") as a:
        ckpt_keys = set(a.keys())
        for path in init.split(","):
            with safe_open(path, "pt") as b:
                init_keys |= set(b.keys())
                for k in sorted(ckpt_keys & set(b.keys())):
                    x, y = a.get_tensor(k).float(), b.get_tensor(k).float()
                    if x.shape != y.shape:
                        continue
                    num += float((x - y).pow(2).sum())
                    den += float(y.pow(2).sum())
                    keys += 1
    return {"compared_tensors": keys, "only_in_ckpt": len(ckpt_keys - init_keys),
            "only_in_init": len(init_keys - ckpt_keys), "rel_l2_change": (num / max(den, 1e-12)) ** 0.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--dit", required=True, help="DiT safetensors to evaluate")
    ap.add_argument("--vae", required=True)
    ap.add_argument("--init", default=None, help="DiT file the run started from (weight-change check)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-quadrant", type=int, default=12)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline
    import lpips

    report = {"dit": args.dit}
    if args.init:
        report["weight_change"] = weight_change(args.dit, args.init)
        print("weight_change", report["weight_change"], flush=True)

    dev = "cuda"
    pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=dev, model_configs=[
        ModelConfig(path=args.dit, offload_device="cpu"), ModelConfig(path=args.vae, offload_device="cpu")])
    pipe.dit.to(dev)
    pipe.vae.to(dev)
    metric = lpips.LPIPS(net="alex").to(dev).eval()

    rng = np.random.default_rng(args.seed)
    rows = [r for r in load_manifest(args.index, "test") if r["num_frames"] >= 60]
    picks = []
    for q in QUADRANTS:
        qr = [r for r in rows if r["quadrant"] == q]
        for i in rng.choice(len(qr), size=min(args.per_quadrant, len(qr)), replace=False):
            r = qr[i]
            s = int(rng.integers(1, r["num_frames"] - 12))
            picks.append((r, [0] + list(range(s, s + 12))))
    window_actions = []
    for r, ids in picks:
        with np.load(r["labels"]) as lab:
            window_actions.append(lab["actions"])

    def gen(imgs, act, seed):
        a = torch.from_numpy(act).unsqueeze(0).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = pipe(seed=seed, tiled=False, input_image=[imgs[0]], input_image4=[imgs[1:5]], action=a,
                       height=256, width=256, num_frames=13, num_inference_steps=args.steps, cfg_scale=1.0,
                       progress_bar_cmd=lambda x: x, batch_size=1)
        return np.stack([np.asarray(im, np.uint8) for im in out[0][5:]])

    def lp(a, b):
        with torch.no_grad():
            return float(metric(to_t(a).to(dev), to_t(b).to(dev)).mean())

    res, t0 = [], time.time()
    for k, (r, ids) in enumerate(picks):
        frames = load_frames(r["agent_segments"], ids, "agentview_image")
        imgs = [Image.fromarray(f) for f in frames]
        real_a = acts(window_actions[k], ids, 1)
        o = picks[(k + len(picks) // 2) % len(picks)]
        other_a = acts(window_actions[(k + len(picks) // 2) % len(picks)], o[1], 1)
        p_real = gen(imgs, real_a, args.seed)
        p_seed = gen(imgs, real_a, args.seed + 1)
        p_other = gen(imgs, other_a, args.seed)
        p_s0 = gen(imgs, acts(window_actions[k], ids, 0), args.seed)
        fut = frames[5:]
        rec = {"idx": r["idx"], "cell": r["cell"], "quadrant": r["quadrant"], "start": ids[1],
               "real": lp(p_real, fut), "copy": lp(np.repeat(frames[4:5], 8, 0), fut), "shift0": lp(p_s0, fut),
               "ctrl": lp(p_real, p_other), "noise": lp(p_real, p_seed)}
        res.append(rec)
        print(f"[{k + 1}/{len(picks)}] {time.time() - t0:.0f}s " + " ".join(f"{x}={rec[x]:.3f}" for x in
              ("real", "copy", "shift0", "ctrl", "noise")), flush=True)

    def agg(rs):
        m = {x: float(np.mean([r[x] for r in rs])) for x in ("real", "copy", "shift0", "ctrl", "noise")}
        m["ctrl_over_noise"] = m["ctrl"] / max(m["noise"], 1e-9)
        m["real_beats_copy_frac"] = float(np.mean([r["real"] < r["copy"] for r in rs]))
        m["shift1_beats_shift0_frac"] = float(np.mean([r["real"] < r["shift0"] for r in rs]))
        m["n"] = len(rs)
        return m

    byq = defaultdict(list)
    for r in res:
        byq[r["quadrant"]].append(r)
    report.update({"all": agg(res), "by_quadrant": {q: agg(v) for q, v in sorted(byq.items())}, "windows": res})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in report if k != "windows"}, indent=1))


if __name__ == "__main__":
    main()
