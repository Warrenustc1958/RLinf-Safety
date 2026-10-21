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

"""Fault-tolerant, resumable Wan AC-SFT trainer (same recipe as train_wan_ac_sft.py).

Differences from the reference loop: counts *optimizer* steps, saves weights +
optimizer state + progress atomically (tmp -> verify -> rename, progress last,
old files deleted only afterwards), resumes from ``<output>/progress.json``,
skips non-finite steps consistently on all ranks, retries failed data reads,
and skips (not crashes) a save when the disk is low. Meant to run under
``run_ft_supervisor.sh``, which relaunches it after any failure.
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

import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

from toolkits.libero_safety.wan_ac_sft.sampler import WanACSFTDataset
from toolkits.libero_safety.wan_ac_sft.train_wan_ac_sft import _load_reference_trainer


class RetryDataset(torch.utils.data.Dataset):
    """Adapts WanACSFTDataset items for the trainer; a failed read retries another index."""

    def __init__(self, inner: WanACSFTDataset, tries: int = 5):
        self.inner, self.tries = inner, tries

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        last = None
        for k in range(self.tries):
            j = (i + k * 7919) % len(self.inner)
            try:
                item = self.inner[j]
                return {"video": item["video"], "reference_image": item["reference_image"],
                        "action": torch.from_numpy(item["action"])}
            except Exception as e:  # NFS hiccup, truncated npz, ...
                last = e
                print(f"[data] item {j} failed ({e!r}), retrying", flush=True)
        raise RuntimeError(f"data item {i} failed {self.tries} times") from last


def atomic_json(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def save_checkpoint(acc, model, opt, out: Path, step: int, args, extra: dict) -> bool:
    """Main process writes weights/optimizer/progress atomically; all ranks wait."""
    acc.wait_for_everyone()
    ok = True
    if acc.is_main_process:
        from safetensors import safe_open
        from safetensors.torch import save_file
        tmp_files = []
        try:
            free = shutil.disk_usage(out).free / 2**30
            if free < args.min_free_gb:
                raise OSError(f"only {free:.0f} GB free (< {args.min_free_gb}); save skipped")
            sd = acc.get_state_dict(model)
            sd = acc.unwrap_model(model).export_trainable_state_dict(sd, remove_prefix="pipe.dit.")
            sd = {k: v.detach().cpu().contiguous() for k, v in sd.items()}
            name, oname = f"step-{step}.safetensors", f"opt-step-{step}.pt"
            wtmp, otmp = out / (name + ".tmp"), out / (oname + ".tmp")
            tmp_files += [wtmp, otmp]
            save_file(sd, str(wtmp))
            with safe_open(str(wtmp), "pt") as f:
                if len(list(f.keys())) != len(sd):
                    raise IOError("weight file verification failed")
            torch.save(opt.state_dict(), otmp)
            os.replace(wtmp, out / name)
            os.replace(otmp, out / oname)
            atomic_json(out / "progress.json", {"opt_step": step, "ckpt": name, "opt": oname,
                                                "saved_at": time.strftime("%F %T"), **extra})
            for p in list(out.glob("step-*.safetensors")) + list(out.glob("opt-step-*.pt")):
                if p.name not in (name, oname):
                    p.unlink()
            print(f"[save] step {step} ok ({free:.0f} GB free before save)", flush=True)
        except Exception as e:
            ok = False
            print(f"[save] step {step} FAILED, training continues: {e!r}", flush=True)
            for p in tmp_files:
                if p.exists():
                    p.unlink()
    acc.wait_for_everyone()
    return ok


def run_val(acc, model, val, n: int) -> float:
    net = acc.unwrap_model(model)
    net.eval()
    tot = torch.zeros(2, device=acc.device)
    with torch.no_grad():
        for i in range(acc.process_index, n, acc.num_processes):
            try:
                loss = net(val[i])
                if torch.isfinite(loss):
                    tot += torch.stack([loss.detach().float(), torch.ones((), device=acc.device)])
            except Exception as e:
                print(f"[val] item {i} failed: {e!r}", flush=True)
    tot = acc.reduce(tot, reduction="sum")
    net.train()
    return float(tot[0] / tot[1]) if tot[1] > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", required=True)
    ap.add_argument("--init_dit", required=True, help="DiT weights for a fresh start")
    ap.add_argument("--vae", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_steps", type=int, required=True, help="optimizer steps")
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--static_video_prob", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=0.0, help="0 = no clipping (norm is only logged)")
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--val_every", type=int, default=5000)
    ap.add_argument("--val_items", type=int, default=200)
    ap.add_argument("--min_free_gb", type=float, default=70)
    ap.add_argument("--max_bad", type=int, default=50, help="abort after this many consecutive bad steps")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--pose_scale", type=float, default=1.0,
                    help="multiply the 6 pose action dims before the model (>1 lifts them to the gripper scale)")
    ap.add_argument("--motion_weight", type=int, default=1,
                    help="1 = draw window starts proportional to the motion profile, 0 = uniform")
    ap.add_argument("--motion_power", type=float, default=1.0)
    ap.add_argument("--motion_floor", type=float, default=0.5,
                    help="floor of the motion weight in 0-255 units; caps down-weighting at 1/motion_floor")
    ap.add_argument("--pose_clip", type=float, default=4.0,
                    help="cap the scaled pose at +-this (0 = no cap); only the fast-motion tail hits it")
    ap.add_argument("--inject_nan_micro", type=int, default=-1, help="testing only")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    prog = json.loads((out / "progress.json").read_text()) if (out / "progress.json").exists() else None
    if prog and prog.get("done"):
        print("[ft] already done", flush=True)
        return

    acc = Accelerator(mixed_precision="bf16",
                      kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=60))])
    ref = _load_reference_trainer()
    dit = str(out / prog["ckpt"]) if prog else args.init_dit
    start = int(prog["opt_step"]) if prog else 0
    acc.print(f"[ft] {'resume' if prog else 'fresh'} from {dit} at opt step {start}/{args.max_steps}, "
              f"world {acc.num_processes}, accum {args.accum}, eff batch {acc.num_processes * args.accum}")

    model = ref.WanTrainingModule(model_paths=json.dumps([[dit], args.vae]), trainable_models="dit",
                                  extra_inputs="input_image,action", static_video_prob=args.static_video_prob,
                                  context_noise_sigma=0.0)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    # val stays uniform so the val loss is comparable across recipes
    train = RetryDataset(WanACSFTDataset(args.index, "train", "agent", 8_000_000, args.seed,
                                         motion_weight=bool(args.motion_weight),
                                         motion_power=args.motion_power,
                                         motion_floor=args.motion_floor,
                                         pose_scale=args.pose_scale,
                                         pose_clip=args.pose_clip))
    val = RetryDataset(WanACSFTDataset(args.index, "val", "agent", args.val_items, args.seed + 1,
                                       motion_weight=False, pose_scale=args.pose_scale,
                                       pose_clip=args.pose_clip))
    gen = torch.Generator()
    gen.manual_seed(args.seed * 1_000_003 + start)  # new data order after every resume
    dl = torch.utils.data.DataLoader(train, sampler=torch.utils.data.RandomSampler(train, generator=gen),
                                     collate_fn=lambda x: x[0], num_workers=args.workers,
                                     persistent_workers=args.workers > 0)
    model, opt, dl = acc.prepare(model, opt, dl)
    if prog and prog.get("opt") and (out / prog["opt"]).exists():
        try:
            opt.load_state_dict(torch.load(out / prog["opt"], map_location="cpu", weights_only=False))
            acc.print("[ft] optimizer state restored")
        except Exception as e:
            acc.print(f"[ft] optimizer state NOT restored ({e!r}); continuing with fresh AdamW")

    stats = (out / "step_stats.jsonl").open("a") if acc.is_main_process else None
    step, bad_run, skipped, micro = start, 0, int(prog.get("skipped", 0)) if prog else 0, 0
    it = iter(dl)
    t = time.perf_counter()
    while step < args.max_steps:
        model.train()
        cycle_bad, losses = False, []
        for m in range(args.accum):
            try:
                data = next(it)
            except StopIteration:
                it = iter(dl)
                data = next(it)
            micro += 1
            ctx = acc.no_sync(model) if m < args.accum - 1 else contextlib.nullcontext()
            with ctx:
                loss = model(data)
                if micro == args.inject_nan_micro:
                    loss = loss * float("nan")
                flag = torch.tensor([0.0 if torch.isfinite(loss).item() else 1.0], device=acc.device)
                if float(acc.reduce(flag, reduction="max")) > 0:  # same decision on every rank
                    cycle_bad = True
                else:
                    acc.backward(loss / args.accum)
                    losses.append(loss.item())
        grad_norm = float("nan")
        if not cycle_bad:
            grad_norm = float(acc.clip_grad_norm_(params, args.grad_clip if args.grad_clip > 0 else float("inf")))
            cycle_bad = not math.isfinite(grad_norm)
        if cycle_bad:
            opt.zero_grad(set_to_none=True)
            bad_run, skipped = bad_run + 1, skipped + 1
            acc.print(f"[ft] non-finite loss/grad at opt step {step + 1}; step skipped ({bad_run} in a row)")
            if bad_run >= args.max_bad:
                raise RuntimeError(f"{bad_run} consecutive non-finite steps")
            continue
        opt.step()
        opt.zero_grad(set_to_none=True)
        step, bad_run = step + 1, 0
        now = time.perf_counter()
        if stats:
            stats.write(json.dumps({"step": step, "loss": sum(losses) / len(losses), "grad_norm": grad_norm,
                                    "sec": now - t, "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}) + "\n")
            stats.flush()
            if step % 100 == 0:
                print(f"[ft] step {step}/{args.max_steps} loss {sum(losses) / len(losses):.4f} "
                      f"grad_norm {grad_norm:.3f} {now - t:.2f}s/step", flush=True)
        t = now
        if args.val_every and step % args.val_every == 0:
            v = run_val(acc, model, val, args.val_items)
            acc.print(f"[val] step {step} loss {v:.5f}")
            if stats:
                stats.write(json.dumps({"step": step, "val_loss": v}) + "\n")
                stats.flush()
        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(acc, model, opt, out, step, args, {
                "skipped": skipped, "max_steps": args.max_steps,
                "pose_scale": args.pose_scale, "motion_weight": args.motion_weight,
                "motion_power": args.motion_power, "motion_floor": args.motion_floor,
                "pose_clip": args.pose_clip})
    if acc.is_main_process:
        p = json.loads((out / "progress.json").read_text())
        if p.get("opt_step") == args.max_steps:
            p["done"] = True
            atomic_json(out / "progress.json", p)
            (out / "DONE").write_text(time.strftime("%F %T") + "\n")
            print("[ft] DONE", flush=True)
        else:
            print("[ft] reached max_steps but final save is missing; supervisor will retry", flush=True)
    acc.wait_for_everyone()


if __name__ == "__main__":
    main()
