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

"""Pipeline check: do actions reach the DiT, and does the output depend on them?

1. Inference path: forward hooks on ``dit.action_mlp1`` / ``action_mlp2`` record
   their inputs; with fixed seed and ``num_inference_steps=1`` the pipeline is
   run with real / zero / other-window / 10x-scaled actions, and output frames
   are compared (mean |diff| in uint8 and LPIPS) against the real-action output.
2. Training path: one ``WanTrainingModule.forward`` per window with the same
   hooks, confirming the action tensor the trainer feeds equals the sampler's.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from toolkits.libero_safety.wan_ac_sft.sampler import WanACSFTDataset, load_frames, load_manifest, window_actions


class Rec:
    def __init__(self):
        self.calls = []

    def hook(self, name):
        def f(_m, inp, _out):
            x = inp[0].detach().float()
            self.calls.append({"layer": name, "shape": list(x.shape), "absmean": float(x.abs().mean()),
                               "sample": x.flatten()[:7].cpu().tolist()})
        return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--dit", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline
    import lpips

    dev = "cuda"
    pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=dev, model_configs=[
        ModelConfig(path=args.dit, offload_device="cpu"), ModelConfig(path=args.vae, offload_device="cpu")])
    pipe.dit.to(dev)
    pipe.vae.to(dev)
    metric = lpips.LPIPS(net="alex").to(dev).eval()
    rec = Rec()
    hooks = [pipe.dit.action_mlp1.register_forward_hook(rec.hook("action_mlp1")),
             pipe.dit.action_mlp2.register_forward_hook(rec.hook("action_mlp2"))]
    flags = {k: getattr(pipe.dit, k, None) for k in ("TI2V1", "TI2V2", "TI2V3", "TI2V4", "five_frame_condition")}

    rng = np.random.default_rng(0)
    rows = [r for r in load_manifest(args.index, "test") if r["num_frames"] >= 60]
    picks = [rows[i] for i in rng.choice(len(rows), size=args.n, replace=False)]
    wins = []
    for r in picks:
        s = int(rng.integers(1, r["num_frames"] - 12))
        ids = [0] + list(range(s, s + 12))
        with np.load(r["labels"]) as lab:
            wins.append((r, ids, window_actions(lab["actions"], ids)))

    def gen(imgs, act, steps=1, seed=0):
        a = torch.from_numpy(act).unsqueeze(0).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = pipe(seed=seed, tiled=False, input_image=[imgs[0]], input_image4=[imgs[1:5]], action=a,
                       height=256, width=256, num_frames=13, num_inference_steps=steps, cfg_scale=1.0,
                       progress_bar_cmd=lambda x: x, batch_size=1)
        return np.stack([np.asarray(im, np.uint8) for im in out[0][5:]])

    def cmp(a, b):
        t = lambda x: torch.from_numpy(x).permute(0, 3, 1, 2).float().to(dev) / 127.5 - 1
        with torch.no_grad():
            lp = float(metric(t(a), t(b)).mean())
        return {"absdiff": float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()), "lpips": lp}

    inf = []
    for k, (r, ids, act) in enumerate(wins):
        frames = load_frames(r["agent_segments"], ids, "agentview_image")
        imgs = [Image.fromarray(f) for f in frames]
        other = wins[(k + 1) % len(wins)][2]
        zero = np.zeros_like(act)
        zero[:, -1] = -1
        rec.calls.clear()
        p_real = gen(imgs, act)
        hook_calls = list(rec.calls)
        res = {"idx": r["idx"], "hook_calls_real": hook_calls[:2],
               "hook_matches_action": bool(hook_calls) and np.allclose(
                   hook_calls[0]["sample"], act.flatten()[:7].tolist(), atol=1e-2),
               "same_action_same_seed": cmp(p_real, gen(imgs, act)),
               "zero_action": cmp(p_real, gen(imgs, zero)),
               "other_action": cmp(p_real, gen(imgs, other)),
               "scaled_10x": cmp(p_real, gen(imgs, np.clip(act * 10, -10, 10))),
               "other_seed": cmp(p_real, gen(imgs, act, seed=1))}
        inf.append(res)
        print(json.dumps({k2: v for k2, v in res.items() if k2 != "hook_calls_real"}), flush=True)
    for h in hooks:
        h.remove()

    # Training path: same hooks on the trainer's DiT, one forward per sample.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ref", "/mnt/nfs/fjj/Code/related_works/diffsynth-studio/examples/wanvideo/model_training/train_rlinf.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    del pipe
    torch.cuda.empty_cache()
    model = ref.WanTrainingModule(model_paths=json.dumps([[args.dit], args.vae]), trainable_models="dit",
                                  extra_inputs="input_image,action", static_video_prob=0.0)
    model.pipe.to(dev)
    model.pipe.device = dev
    rec2 = Rec()
    model.pipe.dit.action_mlp1.register_forward_hook(rec2.hook("action_mlp1"))
    ds = WanACSFTDataset(args.index, "train", "agent", 4, seed=0)
    train = []
    for i in range(4):
        item = ds[i]
        data = {"video": item["video"], "reference_image": item["reference_image"],
                "action": torch.from_numpy(item["action"])}
        rec2.calls.clear()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            loss = float(model(data))
        c = rec2.calls[0] if rec2.calls else None
        train.append({"loss": loss, "hook_called": bool(c), "hook_shape": c["shape"] if c else None,
                      "matches_sampler_action": bool(c) and np.allclose(
                          c["sample"], item["action"].flatten()[:7].tolist(), atol=1e-2)})
        print("train", json.dumps(train[-1]), flush=True)

    def mean(key, sub):
        return float(np.mean([x[key][sub] for x in inf]))
    summary = {"dit_flags": flags,
               "inference_hook_matches_action_frac": float(np.mean([x["hook_matches_action"] for x in inf])),
               "train_hook_matches_frac": float(np.mean([x["matches_sampler_action"] for x in train])),
               **{f"{k}_{s}": mean(k, s) for k in ("same_action_same_seed", "zero_action", "other_action",
                                                    "scaled_10x", "other_seed") for s in ("absdiff", "lpips")}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "inference": inf, "train": train}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
