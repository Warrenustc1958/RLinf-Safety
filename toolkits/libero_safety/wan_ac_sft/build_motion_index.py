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

"""Per-episode motion profile for motion-weighted window sampling.

A training window is ``13`` frames: ``[0] + [s .. s + 11]``. The loss covers the
last two latents, i.e. episode frames ``s + 4 .. s + 11``, so the sampler's
``span`` is 12 and the predicted span of window ``s`` is ``frame[s+4]`` to
``frame[s+11]``.

The profile is the mean absolute pixel change across that span:

    motion[s] = mean | frame[s + 11] - frame[s + 4] |,   s = 1 .. n - 12

in 0-255 units, the same scale as the diagnostic that found 33 % of sampled
windows below 2/255. Written to ``<index>/motion/<idx>.npy`` as float32 of
length ``n + 1`` (index 0 and unusable starts are NaN).

Run under a tmux session; it only reads NFS and needs no GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path("/mnt/nfs/fjj/Code/RLinf-Safety")
sys.path.insert(0, str(REPO))

from toolkits.libero_safety.wan_ac_sft.sampler import load_frames  # noqa: E402

SPAN = 12
FIRST = 4   # offset of the first predicted frame from the window start
LAST = 11   # offset of the last predicted frame


def profile(task) -> tuple[str, int, float]:
    row, out_dir = task
    idx, n = row["idx"], int(row["num_frames"])
    hi = n - SPAN
    out = np.full(n + 1, np.nan, np.float32)
    if hi < 1:
        return idx, 0, float("nan")
    frames = load_frames(row["agent_segments"], list(range(n)), "agentview_image")
    a = frames[1 + FIRST: hi + 1 + FIRST].astype(np.int16)
    b = frames[1 + LAST: hi + 1 + LAST].astype(np.int16)
    d = np.abs(b - a).mean(axis=(1, 2, 3)).astype(np.float32)
    out[1: hi + 1] = d
    dst = Path(out_dir) / f"{idx}.npy"
    tmp = Path(out_dir) / f".{idx}.tmp"          # np.save appends .npy to this name
    with open(tmp, "wb") as f:                   # file object -> no suffix is appended
        np.save(f, out)
    os.replace(tmp, dst)
    return idx, hi, float(np.median(d))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", default="/mnt/nfs/fjj/wan_ac_sft/index/v1")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (smoke test)")
    ap.add_argument("--force", action="store_true", help="rebuild rows that already exist")
    args = ap.parse_args()

    index = Path(args.index)
    out_dir = index / "motion"
    out_dir.mkdir(exist_ok=True)

    rows = [json.loads(l) for l in (index / "manifest.jsonl").read_text().splitlines()]
    wanted = {s.strip() for s in args.splits.split(",")}
    todo = [r for r in rows if r["split"] in wanted
            and (args.force or not (out_dir / f"{r['idx']}.npy").exists())]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[motion] {len(todo)} rows to profile ({len(rows)} in manifest), {args.workers} workers", flush=True)
    if not todo:
        return

    t0, medians, tasks = time.time(), [], [(r, str(out_dir)) for r in todo]
    with Pool(args.workers) as pool:
        for k, (idx, hi, med) in enumerate(pool.imap_unordered(profile, tasks, chunksize=4)):
            medians.append(med)
            if k % 200 == 0:
                print(f"[motion] {k}/{len(todo)} {time.time() - t0:.0f}s last={idx} "
                      f"hi={hi} median={med:.2f}", flush=True)

    m = np.array(medians, np.float32)
    summary = {"rows": len(todo), "seconds": round(time.time() - t0, 1),
               "median_of_medians": float(np.nanmedian(m)),
               "frac_episodes_median_lt2": float(np.nanmean(m < 2))}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print("[motion] done", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
