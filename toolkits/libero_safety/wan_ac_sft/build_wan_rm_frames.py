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

"""Wan-generated reward-model frames with post-success positives.

Recorded episodes stop at the first success, so a successful episode has a single
positive frame. Here the dual Wan replays the episode's own actions over its last
``--window`` frames (anchor = real frame 0, memory = the 4 real frames before the
window, as in training) and then keeps going for ``--hold`` frames with a hold
action (zero pose, last gripper command). Labels: frame f >= first_success -> 1;
hold frames inherit the episode outcome (success -> 1, failure -> 0).

Writes a frame set in the build_frames.py format (``<split>_{agent,wrist}.bin``,
``.off.npy``, ``_meta.npz``, ``_episodes.jsonl``) plus tasks.json copied from the
real frame set, so train_rm.py can mix it in with --extra_train.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from toolkits.libero_safety.wan_ac_sft.sampler import NULL_ACTION, load_frames, load_manifest, scale_pose

QUADRANTS = ("Q1", "Q2", "Q3", "Q4")
SUITES = ("affordance", "human_safety", "obstacle_avoidance", "obstacle_avoidance_human")
SPLITS = {"wantrain": ("cws_v1", "train"), "wannewval": ("cws_test_new", "val"),
          "wannewtest": ("cws_test_new", "test")}


def jpeg(frame, quality=95):
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dit", required=True)
    ap.add_argument("--vae", default="/mnt/nfs/fjj/wan_ac_sft/ckpt/RLinf-Wan-LIBERO-Spatial/Wan2.2_VAE.pth")
    ap.add_argument("--real_frames", default="/data_cf/cws_rm/frames_v1")
    ap.add_argument("--index_root", default="/data_cf/wan_ac_sft/index")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", default="wantrain:1600,wannewval:250,wannewtest:400")
    ap.add_argument("--window", type=int, default=48)
    ap.add_argument("--hold", type=int, default=16)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--pose_scale", type=float, default=8.0)
    ap.add_argument("--pose_clip", type=float, default=4.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(args.real_frames) / "tasks.json", out / "tasks.json")
    tasks = json.loads((out / "tasks.json").read_text())
    inst_id = {s: i for i, s in enumerate(tasks["instructions"])}
    dev = "cuda"

    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline
    from toolkits.libero_safety.wan_ac_sft.dual_view import enable_dual
    pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=dev, model_configs=[
        ModelConfig(path=args.dit, offload_device="cpu"), ModelConfig(path=args.vae, offload_device="cpu")])
    pipe.dit.to(dev)
    pipe.vae.to(dev)
    enable_dual(pipe)

    def chunk(anchor, memory, mem_act, ch, seed):
        a = np.concatenate([NULL_ACTION[None], mem_act, ch], 0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = pipe(seed=seed, tiled=False, input_image=[anchor], input_image4=[memory],
                     action=torch.from_numpy(a).unsqueeze(0).to(dev), height=512, width=256, num_frames=13,
                     num_inference_steps=args.steps, cfg_scale=1.0, progress_bar_cmd=lambda x: x, batch_size=1)
        return [np.asarray(im, np.uint8) for im in o[0][5:]]

    for spec in args.n.split(","):
        split, n = spec.split(":")
        index, isplit = SPLITS[split]
        rows = [r for r in load_manifest(f"{args.index_root}/{index}", isplit)
                if r.get("dual") and r["tcell"] in tasks["tcell_instruction"] and r["num_frames"] >= 14]
        rng = random.Random(0)
        succ = [r for r in rows if r["first_success_frame"] is not None]
        fail = [r for r in rows if r["first_success_frame"] is None]
        rng.shuffle(succ)
        rng.shuffle(fail)
        k = int(n)
        rows = succ[: k * 2 // 3] + fail[: k - min(len(succ), k * 2 // 3)]
        files = {v: open(out / f"{split}_{v}.bin", "wb") for v in ("agent", "wrist")}
        offs = {v: [0] for v in ("agent", "wrist")}
        meta = {key: [] for key in ("episode", "frame", "label", "dist_to_end", "task_id", "quadrant", "suite")}
        t0 = time.time()
        with open(out / f"{split}_episodes.jsonl", "w") as fe:
            e = -1
            for r in rows:
                fs = r["first_success_frame"]
                end = r["num_frames"] - 1 if fs is None else min(fs, r["num_frames"] - 1)
                with np.load(r["labels"]) as lab:
                    acts = lab["actions"].astype(np.float32)
                s = max(5, end - args.window + 1)  # first generated frame
                if end < s:
                    continue
                e += 1  # contiguous episode ids (episode_metrics indexes by position)
                need = list(range(s - 4, s)) + [0]
                ag = load_frames(r["agent_segments"], need, "agentview_image")
                wr = load_frames(r["wrist_segments"], need, "robot0_eye_in_hand_image")
                dual = np.concatenate([ag, wr], 1)
                anchor = Image.fromarray(dual[-1])
                memory = [Image.fromarray(x) for x in dual[:4]]
                mem_act = scale_pose(acts[s - 5: s - 1], args.pose_scale, args.pose_clip)
                hold = np.zeros(7, np.float32)
                hold[6] = acts[end - 1, 6]
                seq = np.concatenate([acts[s - 1: end], np.repeat(hold[None], args.hold, 0)], 0)
                seq = scale_pose(seq, args.pose_scale, args.pose_clip)
                pad = (-len(seq)) % 8
                seq = np.concatenate([seq, np.repeat(seq[-1:], pad, 0)], 0)
                gen = []
                for c in range(len(seq) // 8):
                    ch = seq[8 * c: 8 * c + 8]
                    new = chunk(anchor, memory, mem_act, ch, 1000 * c + e)
                    gen += new
                    memory, mem_act = [Image.fromarray(x) for x in new[-4:]], ch[-4:]
                frames = list(range(s, s + len(gen)))
                keep = [i for i, f in enumerate(frames) if f <= end + args.hold and
                        ((end - f) % args.stride == 0 or f >= end - 8)]
                task = inst_id[tasks["tcell_instruction"][r["tcell"]]]
                for i in keep:
                    f = frames[i]
                    for v, img in (("agent", gen[i][:256]), ("wrist", gen[i][256:])):
                        b = jpeg(img)
                        files[v].write(b)
                        offs[v].append(offs[v][-1] + len(b))
                    meta["episode"].append(e)
                    meta["frame"].append(f)
                    meta["label"].append(fs is not None and f >= fs)
                    meta["dist_to_end"].append(end - f)
                    meta["task_id"].append(task)
                    meta["quadrant"].append(QUADRANTS.index(r["quadrant"]))
                    meta["suite"].append(SUITES.index(r["suite"]))
                fe.write(json.dumps({"episode": e, "idx": r["idx"], "tcell": r["tcell"], "task_id": task,
                                     "quadrant": r["quadrant"], "suite": r["suite"], "level": r["level"],
                                     "success": fs is not None, "num_frames": r["num_frames"],
                                     "num_sampled": len(keep), "window_start": s}) + "\n")
                if (e + 1) % 50 == 0:
                    print(f"[wanrm] {split} {e + 1}/{len(rows)} eps {len(meta['label'])} frames "
                          f"{int(np.sum(meta['label']))} pos {time.time() - t0:.0f}s", flush=True)
        for v in files:
            files[v].close()
            np.save(out / f"{split}_{v}.off.npy", np.asarray(offs[v], np.int64))
        dt = {"episode": np.int32, "frame": np.int32, "label": bool, "dist_to_end": np.int32,
              "task_id": np.int16, "quadrant": np.int8, "suite": np.int8}
        np.savez(out / f"{split}_meta.npz", **{key: np.asarray(v, dt[key]) for key, v in meta.items()})
        print(f"[wanrm] {split} done: {len(rows)} eps {len(meta['label'])} frames "
              f"{int(np.sum(meta['label']))} pos {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
