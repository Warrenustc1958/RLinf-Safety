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

"""Step 4 input: cache real-domain dual-view VAE latents per chunk.

A chunk at t0 is the 13-frame Wan window (schema.chunk_window), stacked agent over
wrist and encoded by the frozen Wan2.2 VAE with the dual patch (each view encoded on
its own). Latent slots 0-1 are z_ctx (anchor, memory), slots 2-3 are z_future.

Selection per role:
  train roles : chunks within +-1 chunk of an event (valid violation or near-miss),
                branch-point chunks (root and branch at t0 = bt-1), plus a random
                ``--train_rest`` fraction of the rest.
  eval roles  : an independent uniform ``--eval_uniform`` sample (flag in_uniform, used
                for the deployment-like main metrics) plus the event / branch chunks.

Writes <out>/<role>_z.npy (fp16 memmap [N, 48, 4, 32, 16]), <out>/<role>_meta.npz
(labels, actions, ids, flags) and <out>/latent_stats.json (cost_train per-channel).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

from toolkits.libero_safety.sam_alignment.schema import (
    CHUNK, K, OUT_ROOT, PROX_CHANNELS, WINDOW, chunk_starts, chunk_window, proximity,
)

VAE = "/mnt/nfs/fjj/wan_ac_sft/ckpt/RLinf-Wan-LIBERO-Spatial/Wan2.2_VAE.pth"
TASKS = "/data_cf/cws_rm/frames_v1/tasks.json"
QUADRANTS = ("Q1", "Q2", "Q3", "Q4")
SUITES = ("affordance", "human_safety", "obstacle_avoidance", "obstacle_avoidance_human")
LATENT = (48, 4, 32, 16)


def load_vae(path: str, device: str):
    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

    from toolkits.libero_safety.wan_ac_sft.dual_view import patch_vae_dual

    pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device,
                                            model_configs=[ModelConfig(path=path, offload_device="cpu")])
    pipe.vae.to(device)
    patch_vae_dual()
    return pipe


@torch.no_grad()
def encode(pipe, frames: np.ndarray, device: str) -> torch.Tensor:
    """uint8 [B, T, 512, 256, 3] -> latents [B, 48, T', 32, 16] (bf16)."""
    x = torch.from_numpy(frames).to(device).permute(0, 4, 1, 2, 3).float().div_(127.5).sub_(1)
    return pipe.vae.single_encode(x.to(torch.bfloat16), device)


def load_dual_frames(ep: dict) -> np.ndarray:
    """Stitched [n, 512, 256, 3] frames (agent over wrist) of an episode."""
    views = []
    for seg_key, key in (("agent_segments", "agentview_image"), ("wrist_segments", "robot0_eye_in_hand_image")):
        parts = []
        for path, a, b in ep[seg_key]:
            with np.load(path) as z:
                parts.append(z[key][a:b])
        views.append(np.concatenate(parts, 0))
    return np.concatenate(views, axis=1)


def select_chunks(eps, roles_of, args, rng):
    """Per role: list of (episode index, t0, in_uniform, is_event, is_pair) and label arrays."""
    pair_t0 = {}
    for e in eps:
        if e["kind"] != "root" and e["root_uid"]:
            pair_t0.setdefault(e["root_uid"], set()).add(int(e["branch_timestep"]) - 1)
    out = {}
    for i, e in enumerate(eps):
        role = roles_of[e["uid"]]
        train = role in args.train_roles
        with np.load(e["labels"]) as z:
            pos = z["cost"] & z["valid_channel"]
            near = z["near_miss"]
        starts = chunk_starts(e["num_frames"])
        ev = np.array([bool(pos[t0:t0 + CHUNK].any() or near[t0:t0 + CHUNK].any()) for t0 in starts])
        nb = ev.copy()
        nb[1:] |= ev[:-1]
        nb[:-1] |= ev[1:]
        pts = set(pair_t0.get(e["uid"], ()))
        if e["kind"] != "root":
            pts.add(int(e["branch_timestep"]) - 1)
        for c, t0 in enumerate(starts):
            is_pair = t0 in pts
            if train:
                keep, uni = nb[c] or is_pair or rng.random() < args.train_rest, False
            else:
                uni = rng.random() < args.eval_uniform
                keep = uni or nb[c] or is_pair
            if keep:
                out.setdefault(role, []).append((i, t0, uni, bool(ev[c]), is_pair))
    return out


def chunk_labels(eps, chunks, task_of):
    n = len(chunks)
    a = {
        "cost": np.zeros((n, CHUNK, K), bool), "valid_step": np.zeros((n, CHUNK), bool),
        "valid_channel": np.zeros((n, CHUNK, K), bool), "near_miss": np.zeros((n, CHUNK, K), bool),
        "prox": np.zeros((n, CHUNK, len(PROX_CHANNELS)), np.float32),
        "prox_valid": np.zeros((n, CHUNK, len(PROX_CHANNELS)), bool),
        "action": np.zeros((n, CHUNK, 7), np.float32), "hazard": np.full((n, CHUNK, K), -1, np.int16),
        "t0": np.zeros(n, np.int32), "episode": np.zeros(n, np.int32), "task_id": np.zeros(n, np.int16),
        "quadrant": np.zeros(n, np.int8), "suite": np.zeros(n, np.int8),
        "in_uniform": np.zeros(n, bool), "is_event": np.zeros(n, bool), "is_pair": np.zeros(n, bool),
    }
    cache = {}
    for r, (i, t0, uni, ev, pair) in enumerate(chunks):
        e = eps[i]
        if i not in cache:
            cache = {i: dict(np.load(e["labels"]))}
        z = cache[i]
        _, vlen = chunk_window(t0, e["num_frames"])
        sl = slice(t0, t0 + vlen)
        a["cost"][r, :vlen] = z["cost"][sl]
        a["valid_step"][r, :vlen] = z["valid_step"][sl]
        a["valid_channel"][r, :vlen] = z["valid_channel"][sl] & z["valid_step"][sl, None]
        a["near_miss"][r, :vlen] = z["near_miss"][sl]
        d = z["distance"][sl][:, PROX_CHANNELS]
        a["prox"][r, :vlen] = np.nan_to_num(proximity(d))
        a["prox_valid"][r, :vlen] = z["distance_valid"][sl][:, PROX_CHANNELS] & ~np.isnan(d)
        a["action"][r, :vlen] = z["action_env"][sl]
        a["hazard"][r, :vlen] = z["hazard"][sl]
        a["t0"][r], a["episode"][r] = t0, i
        a["task_id"][r] = task_of[e["tcell"]]
        a["quadrant"][r] = QUADRANTS.index(e["actual_quadrant"])
        a["suite"][r] = SUITES.index(e["suite"])
        a["in_uniform"][r], a["is_event"][r], a["is_pair"][r] = uni, ev, pair
    return a


def worker(rank, gpus, jobs, eps, out, batch):
    device = f"cuda:{gpus[rank % len(gpus)]}"
    torch.cuda.set_device(device)
    pipe = load_vae(VAE, device)
    mm = {}
    t0 = time.time()
    done = 0
    for n_job, (role, ep_i, rows) in enumerate(jobs[rank]):
        if role not in mm:
            mm[role] = np.lib.format.open_memmap(out / f"{role}_z.npy", mode="r+")
        e = eps[ep_i]
        frames = load_dual_frames(e)
        wins = [chunk_window(t0, e["num_frames"])[0] for _, t0 in rows]
        for s in range(0, len(rows), batch):
            idx = [r for r, _ in rows[s:s + batch]]
            x = np.stack([frames[w] for w in wins[s:s + batch]])
            z = encode(pipe, x, device)
            mm[role][idx] = z.float().cpu().numpy().astype(np.float16)
        done += len(rows)
        if (n_job + 1) % 100 == 0:
            print(f"[w{rank} {device}] {n_job + 1}/{len(jobs[rank])} eps, {done} chunks, "
                  f"{done / (time.time() - t0):.1f} chunk/s", flush=True)
    for m in mm.values():
        m.flush()
    print(f"[w{rank}] done {done} chunks in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", default=str(OUT_ROOT / "labels"))
    ap.add_argument("--registry", default=str(OUT_ROOT / "splits" / "split_registry.json"))
    ap.add_argument("--out", default=str(OUT_ROOT / "cache" / "real_v1"))
    ap.add_argument("--roles", nargs="+", default=["cost_train", "dev", "diag_newtest"])
    ap.add_argument("--train_roles", nargs="+", default=["cost_train", "h_train"])
    ap.add_argument("--train_rest", type=float, default=0.12)
    ap.add_argument("--eval_uniform", type=float, default=0.30)
    ap.add_argument("--gpus", default="0,1,2,3", help="indices into CUDA_VISIBLE_DEVICES")
    ap.add_argument("--procs_per_gpu", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit_eps", type=int, default=0, help="first N episodes per role (pilot)")
    ap.add_argument("--seed", type=int, default=20261024)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reg = json.loads(Path(args.registry).read_text())
    tasks = json.loads(Path(TASKS).read_text())
    inst_id = {s: i for i, s in enumerate(tasks["instructions"])}
    task_of = {tc: inst_id[s] for tc, s in tasks["tcell_instruction"].items()}
    eps = [json.loads(l) for l in (Path(args.labels) / "episodes.jsonl").read_text().splitlines()]
    roles_of = reg["episode_role"]
    eps = [e for e in eps if roles_of.get(e["uid"]) in args.roles and e["label_source"] == "grader_v1" and e["dual"]]
    if args.limit_eps:
        keep, seen = [], {}
        for e in eps:
            r = roles_of[e["uid"]]
            if seen.get(r, 0) < args.limit_eps:
                keep.append(e)
                seen[r] = seen.get(r, 0) + 1
        eps = keep
    rng = np.random.default_rng(args.seed)
    sel = select_chunks(eps, roles_of, args, rng)

    jobs_by_ep = []
    for role in args.roles:
        chunks = sel.get(role, [])
        meta = chunk_labels(eps, chunks, task_of)
        meta["uid"] = np.array([eps[i]["uid"] for i, *_ in chunks])
        meta["group"] = np.array([eps[i]["group"] for i, *_ in chunks])
        np.savez(out / f"{role}_meta.npz", **meta)
        np.lib.format.open_memmap(out / f"{role}_z.npy", mode="w+", dtype=np.float16,
                                  shape=(len(chunks),) + LATENT).flush()
        per_ep = {}
        for r, (i, t0, *_) in enumerate(chunks):
            per_ep.setdefault(i, []).append((r, t0))
        jobs_by_ep += [(role, i, rows) for i, rows in per_ep.items()]
        print(f"{role}: {len(per_ep)} eps, {len(chunks)} chunks "
              f"(uniform {int(meta['in_uniform'].sum())}, event {int(meta['is_event'].sum())}, "
              f"pair {int(meta['is_pair'].sum())}), {len(chunks) * np.prod(LATENT) * 2 / 1e9:.1f} GB", flush=True)

    gpus = [int(g) for g in args.gpus.split(",")]
    nproc = len(gpus) * args.procs_per_gpu
    jobs = [[] for _ in range(nproc)]
    loads = [0] * nproc
    for job in sorted(jobs_by_ep, key=lambda j: -len(j[2])):
        k = int(np.argmin(loads))
        jobs[k].append(job)
        loads[k] += len(job[2])
    (out / "build_args.json").write_text(json.dumps(vars(args), indent=1))
    t0 = time.time()
    mp.spawn(worker, args=(gpus, jobs, eps, out, args.batch), nprocs=nproc, join=True)
    print(f"encoded in {time.time() - t0:.0f}s", flush=True)

    if "cost_train" in args.roles:
        z = np.load(out / "cost_train_z.npy", mmap_mode="r")
        pick = np.sort(rng.choice(len(z), size=min(4000, len(z)), replace=False))
        x = z[pick].astype(np.float32)
        a = np.load(out / "cost_train_meta.npz")
        act = a["action"][a["valid_step"]]
        stats = {"latent_mean": x.mean(axis=(0, 2, 3, 4)).tolist(), "latent_std": x.std(axis=(0, 2, 3, 4)).tolist(),
                 "action_mean": act.mean(0).tolist(), "action_std": (act.std(0) + 1e-6).tolist()}
        (out / "latent_stats.json").write_text(json.dumps(stats, indent=1))
        print("wrote latent_stats.json", flush=True)


if __name__ == "__main__":
    main()
