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

"""Wan AC-SFT trainer for the CWS index (stage 1 + 2 recipe), resumable.

Changes from train_wan_ac_sft_ft.py:
  * fp32 master weights + bf16 autocast. The reference trainer stepped AdamW on bf16
    weights, where an lr-1e-5 update rounds away for most elements;
  * real batches: each sample is VAE-preprocessed on its own, then latents / noise /
    actions are concatenated and the DiT runs once per batch (loss checked equal to
    the per-sample mean). The diffusion timestep is shared within a batch, as in the
    reference loss;
  * per-rank RNG seeding (torch's default seed is the same in every process, so the
    reference trainer drew the same timestep and noise on every rank);
  * gradient clipping, linear warmup + cosine decay, EMA of the weights (rank 0);
  * fixed validation set: the same windows, timesteps and noise at every evaluation,
    reported per motion bin of the predicted span.
Checkpoints: step-<k>.safetensors (raw, latest only), ema-step-<k>.safetensors (bf16,
milestones kept), resume/ (fp32 weights, optimizer, EMA; latest only), progress.json.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs

from toolkits.libero_safety.wan_ac_sft.fp32_master import patch_flash_attention, to_fp32_master
from toolkits.libero_safety.wan_ac_sft.sampler import WanACSFTDataset
from toolkits.libero_safety.wan_ac_sft.train_wan_ac_sft import _load_reference_trainer

CAT_KEYS = ("noise", "latents", "input_latents", "first_frame_latents")
PREFIX = "pipe.dit."
BINS = ("template", "lt2", "2to5", "ge5")


class ItemDataset(torch.utils.data.Dataset):
    """WanACSFTDataset item -> trainer dict; a failed read retries another index."""

    def __init__(self, inner, tries=5):
        self.inner, self.tries = inner, tries

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        last = None
        for k in range(self.tries):
            j = (i + k * 7919) % len(self.inner)
            try:
                it = self.inner[j]
                return {"video": it["video"], "reference_image": it["reference_image"],
                        "action": torch.from_numpy(it["action"])}
            except Exception as e:  # NFS hiccup, truncated npz, ...
                last = e
                print(f"[data] item {j} failed ({e!r}), retrying", flush=True)
        raise RuntimeError(f"data item {i} failed {self.tries} times") from last


class BatchLoss(torch.nn.Module):
    """Per-sample preprocessing (VAE, frozen) then one DiT forward over the batch."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, items):
        with torch.no_grad():
            inps = [self.net.forward_preprocess(d) for d in items]
        x = dict(inps[0])
        for k in CAT_KEYS:
            x[k] = torch.cat([p[k] for p in inps], 0)
        x["action"] = torch.stack([p["action"] for p in inps], 0)
        models = {n: getattr(self.net.pipe, n) for n in self.net.pipe.in_iteration_models}
        return self.net.pipe.training_loss(**models, **x)


def lr_factor(step, warmup, total, floor):
    if step < warmup:
        return (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))


def motion_bin(index_dir, row, frame_ids):
    if frame_ids[1] == 0:
        return "template"
    p = Path(index_dir) / "motion" / f"{row['idx']}.npy"
    if not p.exists():
        return "2to5"
    m = np.load(p)
    v = m[frame_ids[1]] if frame_ids[1] < len(m) else float("nan")
    if not np.isfinite(v):
        return "template"
    return "lt2" if v < 2 else ("2to5" if v < 5 else "ge5")


def atomic_json(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", required=True)
    ap.add_argument("--init_dit", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_steps", type=int, required=True)
    ap.add_argument("--batch", type=int, default=20, help="samples per GPU per micro step")
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr_floor", type=float, default=0.1, help="final lr / peak lr of the cosine")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--static_video_prob", type=float, default=0.05)
    ap.add_argument("--pose_scale", type=float, default=8.0)
    ap.add_argument("--pose_clip", type=float, default=4.0)
    ap.add_argument("--motion_weight", type=int, default=1)
    ap.add_argument("--motion_floor", type=float, default=0.5)
    ap.add_argument("--motion_power", type=float, default=1.0)
    ap.add_argument("--save_every", type=int, default=2500)
    ap.add_argument("--ema_keep_every", type=int, default=10000, help="keep EMA exports at multiples of this")
    ap.add_argument("--val_every", type=int, default=2500)
    ap.add_argument("--val_items", type=int, default=400)
    ap.add_argument("--min_free_gb", type=float, default=110)
    ap.add_argument("--save_raw", type=int, default=1, help="0 = do not export the raw (non-EMA) bf16 weights")
    ap.add_argument("--resume_inplace", type=int, default=0,
                    help="1 = overwrite resume/ in place (peak +~10 GB instead of +~85 GB); a crash mid-save "
                         "falls back to the newest EMA export with a fresh optimizer")
    ap.add_argument("--max_bad", type=int, default=50)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    prog = json.loads((out / "progress.json").read_text()) if (out / "progress.json").exists() else None
    if prog and prog.get("done"):
        print("[cws] already done", flush=True)
        return
    start = int(prog["opt_step"]) if prog else 0
    fallback = None  # EMA export to restart from when resume/ is incomplete
    if prog:
        rdir, mark = out / "resume", out / "resume" / "COMPLETE"
        has = (rdir / "model_fp32.pt").exists() and (rdir / "opt.pt").exists()
        ok = has and ((mark.exists() and mark.read_text().strip() == str(start))
                      or (not mark.exists() and not prog.get("resume_inplace")))  # legacy atomic save
        if not ok:
            emas = sorted(out.glob("ema-step-*.safetensors"), key=lambda p: int(p.stem.split("-")[-1]))
            if not emas:
                raise RuntimeError("resume/ is incomplete and there is no EMA export to fall back to")
            fallback = emas[-1]
            start = int(fallback.stem.split("-")[-1])
            print(f"[cws] resume/ incomplete; restarting from {fallback.name} with a fresh optimizer", flush=True)

    acc = Accelerator(mixed_precision="bf16", kwargs_handlers=[
        InitProcessGroupKwargs(timeout=timedelta(minutes=60)),
        DistributedDataParallelKwargs(gradient_as_bucket_view=True, find_unused_parameters=False)])
    torch.manual_seed(args.seed * 1_000_003 + 7919 * acc.process_index + start)
    np.random.seed((args.seed * 7919 + 104729 * acc.process_index + start) % 2**32)

    ref = _load_reference_trainer()
    patch_flash_attention()
    net = ref.WanTrainingModule(model_paths=json.dumps([[args.init_dit], args.vae]), trainable_models="dit",
                                extra_inputs="input_image,action", static_video_prob=args.static_video_prob,
                                context_noise_sigma=0.0)
    net.pipe.device = acc.device
    named = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
    if prog and fallback is None:  # fp32 weights of the resume point
        sd = torch.load(out / "resume" / "model_fp32.pt", map_location="cpu", weights_only=True)
        with torch.no_grad():
            for n, p in named:
                p.data = sd[n].to(p.device)
        del sd
    elif fallback is not None:
        from safetensors.torch import load_file
        sd = load_file(str(fallback))
        with torch.no_grad():
            for n, p in named:
                p.data = sd[n[len(PREFIX):] if n.startswith(PREFIX) else n].to(p.device)
        del sd
    to_fp32_master(net)
    params = [p for _, p in named]
    acc.print(f"[cws] {'resume' if prog else 'fresh'} at opt step {start}/{args.max_steps}; world "
              f"{acc.num_processes} x batch {args.batch} x accum {args.accum} = eff batch "
              f"{acc.num_processes * args.batch * args.accum}; trainable {sum(p.numel() for p in params) / 1e9:.2f}B fp32")

    model = BatchLoss(net)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: lr_factor(k + start, args.warmup, args.max_steps, args.lr_floor))

    train = ItemDataset(WanACSFTDataset(args.index, "train", "agent", 50_000_000, args.seed,
                                        motion_weight=bool(args.motion_weight), motion_power=args.motion_power,
                                        motion_floor=args.motion_floor, pose_scale=args.pose_scale,
                                        pose_clip=args.pose_clip))
    gen = torch.Generator()
    gen.manual_seed(args.seed * 1_000_003 + start)
    dl = torch.utils.data.DataLoader(train, batch_size=args.batch,
                                     sampler=torch.utils.data.RandomSampler(train, generator=gen),
                                     collate_fn=lambda x: x, num_workers=args.workers,
                                     persistent_workers=args.workers > 0, prefetch_factor=4, drop_last=True)
    model, opt, dl = acc.prepare(model, opt, dl)
    if prog and fallback is None and (out / "resume" / "opt.pt").exists():
        opt.load_state_dict(torch.load(out / "resume" / "opt.pt", map_location="cpu", weights_only=False))
        acc.print("[cws] optimizer state restored")

    # EMA sharded across ranks: rank r tracks tensors i with i % world == r
    mine = [i for i in range(len(params)) if i % acc.num_processes == acc.process_index]
    ema = None
    if args.ema > 0:
        shards = sorted((out / "resume").glob("ema_shard_*.pt")) if prog and fallback is None else []
        if shards:
            want, got = {named[i][0] for i in mine}, {}
            for f in shards:
                e = torch.load(f, map_location="cpu", weights_only=True)
                got.update({k: v for k, v in e.items() if k in want})
                del e
            ema = [got[named[i][0]].to(acc.device) for i in mine]
            del got
        else:
            ema = [params[i].detach().clone() for i in mine]
    ema_params = [params[i] for i in mine]

    # fixed validation set (uniform windows, held-out init states)
    val_ds = WanACSFTDataset(args.index, "val", "agent", args.val_items, args.seed + 1, motion_weight=False,
                             pose_scale=args.pose_scale, pose_clip=args.pose_clip)
    val_bins = []
    for i in range(args.val_items):
        row, ids, _ = val_ds.sampler.draw(np.random.default_rng([val_ds.seed, i]))
        val_bins.append(motion_bin(args.index, row, ids))

    def run_val():
        inner = acc.unwrap_model(model)
        inner.eval()
        tot = torch.zeros(2 * (len(BINS) + 1), device=acc.device)
        with torch.no_grad():
            for i in range(acc.process_index, args.val_items, acc.num_processes):
                it = val_ds[i]
                d = {"video": it["video"], "reference_image": it["reference_image"],
                     "action": torch.from_numpy(it["action"])}
                with torch.random.fork_rng(devices=[acc.device]):
                    torch.manual_seed(1_000_000 + i)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = inner([d])
                if torch.isfinite(loss):
                    b = BINS.index(val_bins[i])
                    tot[2 * b] += loss.float(); tot[2 * b + 1] += 1
                    tot[-2] += loss.float(); tot[-1] += 1
        tot = acc.reduce(tot, reduction="sum")
        inner.train()
        res = {name: (float(tot[2 * k] / tot[2 * k + 1]) if tot[2 * k + 1] > 0 else None)
               for k, name in enumerate(BINS)}
        res["all"] = float(tot[-2] / tot[-1]) if tot[-1] > 0 else None
        res["n"] = {name: int(tot[2 * k + 1]) for k, name in enumerate(BINS)}
        mem = torch.tensor([torch.cuda.max_memory_allocated() / 2**30], device=acc.device)
        res["peak_mem_gb_per_rank"] = [round(float(x), 1) for x in acc.gather(mem)]
        return res

    def export(named_tensors, path):
        """{full param name: tensor} -> bf16 safetensors with the reference key names."""
        from safetensors.torch import save_file
        sd = {n[len(PREFIX):] if n.startswith(PREFIX) else n: t.detach().to(torch.bfloat16).cpu().contiguous()
              for n, t in named_tensors.items()}
        tmp = Path(str(path) + ".tmp")
        save_file(sd, str(tmp))
        os.replace(tmp, path)

    def save(step):
        acc.wait_for_everyone()
        tmp = out / ("resume" if args.resume_inplace else "resume.tmp")
        ok = torch.ones(1, device=acc.device)
        if acc.is_main_process:
            free = shutil.disk_usage(out).free / 2**30
            if free < args.min_free_gb:
                print(f"[save] step {step} skipped: only {free:.0f} GB free (< {args.min_free_gb})", flush=True)
                ok[0] = 0
            else:
                shutil.rmtree(tmp, ignore_errors=True)
                tmp.mkdir()
        if float(acc.reduce(ok, reduction="min")) == 0:
            acc.wait_for_everyone()
            return
        acc.wait_for_everyone()
        if ema is not None:  # every rank writes its EMA shard
            torch.save({named[i][0]: e.detach().cpu() for i, e in zip(mine, ema)},
                       tmp / f"ema_shard_{acc.process_index}.pt")
        acc.wait_for_everyone()
        if acc.is_main_process:
            try:
                free = shutil.disk_usage(out).free / 2**30
                if args.save_raw:
                    export({n: p for n, p in named}, out / f"step-{step}.safetensors")
                if ema is not None:
                    merged = {}
                    for f in sorted(tmp.glob("ema_shard_*.pt")):
                        merged.update(torch.load(f, map_location="cpu", weights_only=True))
                    if len(merged) != len(named):
                        raise IOError(f"EMA shards hold {len(merged)} tensors, expected {len(named)}")
                    export(merged, out / f"ema-step-{step}.safetensors")
                    del merged
                torch.save({n: p.detach().cpu() for n, p in named}, tmp / "model_fp32.pt")
                torch.save(opt.state_dict(), tmp / "opt.pt")
                (tmp / "COMPLETE").write_text(str(step))
                old = out / "resume.old"
                if not args.resume_inplace:
                    shutil.rmtree(old, ignore_errors=True)
                    if (out / "resume").exists():
                        os.replace(out / "resume", old)
                    os.replace(tmp, out / "resume")
                atomic_json(out / "progress.json", {"opt_step": step,
                                                    "ckpt": f"step-{step}.safetensors" if args.save_raw else None,
                                                    "ema": f"ema-step-{step}.safetensors",
                                                    "resume_inplace": bool(args.resume_inplace),
                                                    "saved_at": time.strftime("%F %T"), "args": vars(args)})
                shutil.rmtree(old, ignore_errors=True)
                for p in out.glob("step-*.safetensors"):
                    if not args.save_raw or p.name != f"step-{step}.safetensors":
                        p.unlink()
                for p in out.glob("ema-step-*.safetensors"):  # ema_keep_every <= 0: latest only
                    k = int(p.stem.split("-")[-1])
                    if k != step and (args.ema_keep_every <= 0 or k % args.ema_keep_every != 0):
                        p.unlink()
                print(f"[save] step {step} ok ({free:.0f} GB free before)", flush=True)
            except Exception as e:
                print(f"[save] step {step} FAILED, training continues: {e!r}", flush=True)
        acc.wait_for_everyone()

    stats = (out / "step_stats.jsonl").open("a") if acc.is_main_process else None
    step, bad_run, it_dl = start, 0, iter(dl)
    t = time.perf_counter()
    while step < args.max_steps:
        model.train()
        cycle_bad, losses = False, []
        for m in range(args.accum):
            try:
                batch = next(it_dl)
            except StopIteration:
                it_dl = iter(dl)
                batch = next(it_dl)
            ctx = acc.no_sync(model) if m < args.accum - 1 else contextlib.nullcontext()
            with ctx:
                loss = model(batch)
                flag = torch.tensor([0.0 if torch.isfinite(loss).item() else 1.0], device=acc.device)
                if float(acc.reduce(flag, reduction="max")) > 0:
                    cycle_bad = True
                else:
                    acc.backward(loss / args.accum)
                    losses.append(loss.item())
        gn = float("nan")
        if not cycle_bad:
            gn = float(acc.clip_grad_norm_(params, args.grad_clip if args.grad_clip > 0 else float("inf")))
            cycle_bad = not math.isfinite(gn)
        if cycle_bad:
            opt.zero_grad(set_to_none=False)
            bad_run += 1
            acc.print(f"[cws] non-finite loss/grad at step {step + 1}, skipped ({bad_run} in a row)")
            if bad_run >= args.max_bad:
                raise RuntimeError(f"{bad_run} consecutive non-finite steps")
            continue
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=False)
        step, bad_run = step + 1, 0
        if ema is not None:
            d = min(args.ema, (1 + step) / (10 + step))
            with torch.no_grad():
                torch._foreach_mul_(ema, d)
                torch._foreach_add_(ema, [p.detach() for p in ema_params], alpha=1 - d)
        now = time.perf_counter()
        if stats:
            stats.write(json.dumps({"step": step, "loss": sum(losses) / len(losses), "grad_norm": gn,
                                    "lr": sched.get_last_lr()[0], "sec": now - t,
                                    "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}) + "\n")
            stats.flush()
            if step % 50 == 0:
                print(f"[cws] step {step}/{args.max_steps} loss {sum(losses) / len(losses):.4f} gn {gn:.3f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {now - t:.2f}s/step", flush=True)
        t = now
        if args.val_every and step % args.val_every == 0:
            v = run_val()
            acc.print(f"[val] step {step} " + json.dumps(v))
            if stats:
                stats.write(json.dumps({"step": step, "val": v}) + "\n")
                stats.flush()
        if step % args.save_every == 0 or step == args.max_steps:
            save(step)
    if acc.is_main_process:
        p = json.loads((out / "progress.json").read_text())
        if p.get("opt_step") == args.max_steps:
            p["done"] = True
            atomic_json(out / "progress.json", p)
            (out / "DONE").write_text(time.strftime("%F %T") + "\n")
            print("[cws] DONE", flush=True)
    acc.wait_for_everyone()


if __name__ == "__main__":
    main()
