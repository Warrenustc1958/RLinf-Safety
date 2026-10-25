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

"""Step 5: real / generated latent pairs from the frozen dual-view Wan.

Modes (one row = one 8-step chunk, labels = the real executed outcome):
  0 tf    teacher-forced: real anchor + real memory + recorded actions; rows are a
          subset of the real cache (pairs, events, uniform), real latent copied.
  1 ar    autoregressive from frame 0 with the recorded actions; memory = last 4
          generated frames; one rollout per episode, depth = chunk index + 1; the
          real reference latent is encoded from the same 13-frame real window.
  2 seed  tf rows regenerated with other seeds (stability of generated danger).

The generated latent is captured right before VAE decoding (slots 0-1 = the
context latents the pipeline conditions on, 2-3 = generated future), so it is the
latent Wan actually produced, not a re-encoding of decoded frames.
Writes <out>/{gen_z,real_z}.npy (fp16 [N,48,4,32,16]), meta.npz, sheets/*.png.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image

from toolkits.libero_safety.sam_alignment.cache_real import (
    LATENT, TASKS, chunk_labels, encode, load_dual_frames,
)
from toolkits.libero_safety.sam_alignment.schema import CHUNK, OUT_ROOT, chunk_starts, chunk_window
from toolkits.libero_safety.wan_ac_sft.sampler import NULL_ACTION, window_actions

DIT = "/data_cf/wan_ac_sft/runs/s3_dual_b10/ema-step-41000.safetensors"
VAE = "/mnt/nfs/fjj/wan_ac_sft/ckpt/RLinf-Wan-LIBERO-Spatial/Wan2.2_VAE.pth"
MODES = ("tf", "ar", "seed")


def load_wan(device):
    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

    from toolkits.libero_safety.wan_ac_sft.dual_view import enable_dual

    pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=[
        ModelConfig(path=DIT, offload_device="cpu"), ModelConfig(path=VAE, offload_device="cpu")])
    pipe.dit.to(device)
    pipe.vae.to(device)
    enable_dual(pipe)
    store = {}
    orig = pipe.vae.decode

    def decode(hidden_states, *a, **k):  # capture the final latent before decoding
        store["z"] = hidden_states.detach()
        return orig(hidden_states, *a, **k)

    pipe.vae.decode = decode
    return pipe, store


def generate(pipe, store, anchor, memory, actions, seed, steps, device):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = pipe(seed=seed, tiled=False, input_image=[Image.fromarray(anchor)],
                   input_image4=[[Image.fromarray(m) for m in memory]],
                   action=torch.from_numpy(actions).unsqueeze(0).to(device), height=512, width=256, num_frames=13,
                   num_inference_steps=steps, cfg_scale=1.0, progress_bar_cmd=lambda x: x, batch_size=1)
    z = store["z"][0]
    assert tuple(z.shape) == LATENT, z.shape
    return np.stack([np.asarray(im, np.uint8) for im in out[0][5:]]), z.float().cpu().numpy().astype(np.float16)


def sheet(path, real, gen):
    """Top: real future (agent | wrist), bottom: generated, 8 frames, 64 px per view."""
    def strip(fr):
        return np.concatenate([np.asarray(Image.fromarray(f).resize((64, 128))) for f in fr], 1)
    Image.fromarray(np.concatenate([strip(real), strip(gen)], 0)).save(path)


def worker(rank, gpus, jobs, eps, out, args):
    device = f"cuda:{gpus[rank % len(gpus)]}"
    torch.cuda.set_device(device)
    pipe, store = load_wan(device)
    gen_mm = np.lib.format.open_memmap(out / "gen_z.npy", mode="r+")
    real_mm = np.lib.format.open_memmap(out / "real_z.npy", mode="r+")
    t0_wall, done = time.time(), 0
    for n_job, job in enumerate(jobs[rank]):
        e = eps[job["ep"]]
        frames = load_dual_frames(e)
        with np.load(e["labels"]) as z:
            acts = z["action_env"].astype(np.float32)
        n = e["num_frames"]
        for row, t0, seed, mk_sheet in job["tf"]:
            ids, _ = chunk_window(t0, n)
            a = window_actions(acts, ids, args.pose_scale, args.pose_clip)
            gen, zg = generate(pipe, store, frames[0], frames[ids[1:5]], a, seed, args.steps, device)
            gen_mm[row] = zg
            if mk_sheet:
                sheet(out / "sheets" / f"row{row:06d}.png", frames[ids[5:]], gen)
            done += 1
        if job["ar"]:
            memory, mem_act = np.stack([frames[0]] * 4), np.repeat(NULL_ACTION[None], 4, 0)
            for c, (row, mk_sheet) in enumerate(job["ar"]):
                t0 = CHUNK * c
                ids, _ = chunk_window(t0, n)
                fut = window_actions(acts, ids, args.pose_scale, args.pose_clip)[5:]
                a = np.concatenate([NULL_ACTION[None], mem_act, fut], 0)
                gen, zg = generate(pipe, store, frames[0], memory, a, 1000 * c + job["ep"], args.steps, device)
                gen_mm[row] = zg
                real_mm[row] = encode(pipe, frames[ids][None], device)[0].float().cpu().numpy().astype(np.float16)
                if mk_sheet:
                    sheet(out / "sheets" / f"row{row:06d}.png", frames[ids[5:]], gen)
                memory, mem_act = gen[-4:], fut[-4:]
                done += 1
        if (n_job + 1) % 10 == 0:
            print(f"[w{rank} {device}] {n_job + 1}/{len(jobs[rank])} eps {done} chunks "
                  f"{(time.time() - t0_wall) / max(done, 1):.2f} s/chunk", flush=True)
    gen_mm.flush()
    real_mm.flush()
    print(f"[w{rank}] done {done} chunks in {time.time() - t0_wall:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", default=str(OUT_ROOT / "labels"))
    ap.add_argument("--registry", default=str(OUT_ROOT / "splits" / "split_registry.json"))
    ap.add_argument("--real_cache", default=str(OUT_ROOT / "cache" / "real_v1"))
    ap.add_argument("--role", default="dev")
    ap.add_argument("--out", default=str(OUT_ROOT / "pairs" / "pilot_v1"))
    ap.add_argument("--tf_pair", type=int, default=100000, help="max teacher-forced pair rows")
    ap.add_argument("--tf_event", type=int, default=800)
    ap.add_argument("--tf_uniform", type=int, default=600)
    ap.add_argument("--ar_eps", type=int, default=60)
    ap.add_argument("--ar_depth", type=int, default=16)
    ap.add_argument("--seed_rows", type=int, default=200)
    ap.add_argument("--extra_seeds", type=int, default=2)
    ap.add_argument("--sheets", type=int, default=40)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--pose_scale", type=float, default=8.0)
    ap.add_argument("--pose_clip", type=float, default=4.0)
    ap.add_argument("--gpus", default="0,1,2,3,4")
    ap.add_argument("--procs_per_gpu", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261024)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    reg = json.loads(Path(args.registry).read_text())
    roles = reg["episode_role"]
    tasks = json.loads(Path(TASKS).read_text())
    inst = {s: i for i, s in enumerate(tasks["instructions"])}
    task_of = {tc: inst[s] for tc, s in tasks["tcell_instruction"].items()}
    eps = [json.loads(l) for l in (Path(args.labels) / "episodes.jsonl").read_text().splitlines()]
    eps = [e for e in eps if roles.get(e["uid"]) == args.role and e["label_source"] == "grader_v1" and e["dual"]]
    ep_of = {e["uid"]: i for i, e in enumerate(eps)}

    # ---- teacher-forced rows: subset of the real cache rows of this role
    rm = np.load(Path(args.real_cache) / f"{args.role}_meta.npz")
    pair_rows = np.flatnonzero(rm["is_pair"])
    pair_rows = rng.permutation(pair_rows)[: args.tf_pair]
    ev_rows = np.setdiff1d(np.flatnonzero(rm["is_event"]), pair_rows)
    ev_rows = rng.permutation(ev_rows)[: args.tf_event]
    un_rows = np.setdiff1d(np.flatnonzero(rm["in_uniform"]), np.concatenate([pair_rows, ev_rows]))
    un_rows = rng.permutation(un_rows)[: args.tf_uniform]
    tf_src = np.sort(np.concatenate([pair_rows, ev_rows, un_rows]))
    seed_src = rng.permutation(np.intersect1d(tf_src, np.flatnonzero(rm["is_event"])))[: args.seed_rows]

    chunks, mode, depth, seed, src = [], [], [], [], []
    for r in tf_src:
        i = ep_of[str(rm["uid"][r])]
        chunks.append((i, int(rm["t0"][r]), bool(rm["in_uniform"][r]), bool(rm["is_event"][r]), bool(rm["is_pair"][r])))
        mode.append(0); depth.append(0); seed.append(int(r)); src.append(int(r))
    for k in range(1, args.extra_seeds + 1):
        for r in seed_src:
            i = ep_of[str(rm["uid"][r])]
            chunks.append((i, int(rm["t0"][r]), bool(rm["in_uniform"][r]), True, bool(rm["is_pair"][r])))
            mode.append(2); depth.append(0); seed.append(int(r) + 100000 * k); src.append(int(r))

    # ---- autoregressive episodes: stratified by actual quadrant, long enough, prefer events
    long_eps = [i for i, e in enumerate(eps) if e["num_frames"] - 1 >= CHUNK * 8]
    quota = {"Q4": 0.35, "Q2": 0.25, "Q3": 0.15, "Q1": 0.25}
    ar_eps = []
    for q, f in quota.items():
        cand = [i for i in long_eps if eps[i]["actual_quadrant"] == q]
        cand.sort(key=lambda i: (eps[i]["kind"] == "root", rng.random()))
        ar_eps += cand[: int(round(f * args.ar_eps))]
    ar_rows_of = {}
    for i in ar_eps:
        e = eps[i]
        with np.load(e["labels"]) as z:
            pos, near = z["cost"] & z["valid_channel"], z["near_miss"]
        starts = chunk_starts(e["num_frames"])[: args.ar_depth]
        ar_rows_of[i] = []
        for c, t0 in enumerate(starts):
            ev = bool(pos[t0:t0 + CHUNK].any() or near[t0:t0 + CHUNK].any())
            ar_rows_of[i].append(len(chunks))
            chunks.append((i, t0, False, ev, False))
            mode.append(1); depth.append(c + 1); seed.append(1000 * c + i); src.append(-1)

    meta = chunk_labels(eps, chunks, task_of)
    meta.update(mode=np.array(mode, np.int8), depth=np.array(depth, np.int8), gen_seed=np.array(seed, np.int64),
                src_row=np.array(src, np.int64), uid=np.array([eps[c[0]]["uid"] for c in chunks]),
                group=np.array([eps[c[0]]["group"] for c in chunks]),
                root_uid=np.array([eps[c[0]]["root_uid"] or eps[c[0]]["uid"] for c in chunks]),
                kind=np.array([eps[c[0]]["kind"] for c in chunks]))
    np.savez(out / "meta.npz", **meta)
    N = len(chunks)
    np.lib.format.open_memmap(out / "gen_z.npy", mode="w+", dtype=np.float16, shape=(N,) + LATENT).flush()
    real_mm = np.lib.format.open_memmap(out / "real_z.npy", mode="w+", dtype=np.float16, shape=(N,) + LATENT)
    rz = np.load(Path(args.real_cache) / f"{args.role}_z.npy", mmap_mode="r")
    has_src = np.flatnonzero(meta["src_row"] >= 0)
    for s in range(0, len(has_src), 2048):
        rows = has_src[s:s + 2048]
        real_mm[rows] = rz[meta["src_row"][rows]]
    real_mm.flush()
    del real_mm

    sheet_rows = set(rng.permutation(np.flatnonzero((meta["mode"] == 0) & meta["is_event"]))[: args.sheets // 2].tolist())
    for i in ar_eps[:4]:
        sheet_rows |= {r for c, r in enumerate(ar_rows_of[i]) if c + 1 in (1, 4, 8, 16)}
    jobs_by_ep = {}
    for r in range(N):
        i = chunks[r][0]
        j = jobs_by_ep.setdefault(i, {"ep": i, "tf": [], "ar": []})
        if mode[r] in (0, 2):
            j["tf"].append((r, chunks[r][1], seed[r], r in sheet_rows))
    for i, rows in ar_rows_of.items():
        jobs_by_ep.setdefault(i, {"ep": i, "tf": [], "ar": []})["ar"] = [(r, r in sheet_rows) for r in rows]
    gpus = [int(g) for g in args.gpus.split(",")]
    nproc = len(gpus) * args.procs_per_gpu
    jobs, loads = [[] for _ in range(nproc)], [0] * nproc
    for j in sorted(jobs_by_ep.values(), key=lambda j: -(len(j["tf"]) + len(j["ar"]))):
        k = int(np.argmin(loads))
        jobs[k].append(j)
        loads[k] += len(j["tf"]) + len(j["ar"])
    (out / "build_args.json").write_text(json.dumps(vars(args), indent=1))
    print(f"rows {N}: tf {int((meta['mode'] == 0).sum())} (pair {len(pair_rows)}, event {len(ev_rows)}, "
          f"uniform {len(un_rows)}), seed {int((meta['mode'] == 2).sum())}, ar {int((meta['mode'] == 1).sum())} "
          f"({len(ar_eps)} eps); {N * np.prod(LATENT) * 4 / 1e9:.1f} GB; workers {nproc}", flush=True)
    t0 = time.time()
    mp.spawn(worker, args=(gpus, jobs, eps, out, args), nprocs=nproc, join=True)
    print(f"generated in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
