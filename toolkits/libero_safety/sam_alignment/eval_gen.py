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

"""Step 5 pilot evaluation: frozen real-domain SAM baselines on generated latents.

For the same rows (labels = real executed outcome) compares B2 on the real future
latent vs the Wan-generated one, B1 (context only) and B0 (task + action), per
mode (teacher-forced / autoregressive depth buckets). Also reports
  * miss rate at the dev-fixed R90 threshold (generated false-safe),
  * same-state counterfactual ordering on root / branch pairs,
  * seed stability of generated danger (seed rows vs their tf row),
  * latent fidelity (cosine of generated vs real future, agent / wrist).
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch

from toolkits.libero_safety.sam_alignment.model import SAMNet
from toolkits.libero_safety.sam_alignment.schema import CHANNELS, OUT_ROOT
from toolkits.libero_safety.sam_alignment.train import auroc_ap, chunk_labels

DEPTH_BUCKETS = {"d1": (1, 1), "d2": (2, 2), "d3-4": (3, 4), "d5-8": (5, 8), "d9-16": (9, 16)}


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = SAMNet(**ck["config"])
    m.load_state_dict(ck["model_state_dict"])
    return m.cuda().eval()


@torch.no_grad()
def logits(model, z_mm, meta, bs=256):
    out = []
    for s in range(0, len(meta["t0"]), bs):
        z = torch.from_numpy(np.asarray(z_mm[s:s + bs], dtype=np.float32)).cuda()
        a = torch.from_numpy(meta["action"][s:s + bs]).cuda()
        t = torch.from_numpy(meta["task_id"][s:s + bs].astype(np.int64)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, _ = model(z, a, t)
        out.append(lg.float().cpu().numpy())
    return np.concatenate(out)


def chunk_ap(meta, rows, lg, thr=None):
    pos, known, score, _ = chunk_labels(meta, rows, lg[rows])
    res = {}
    for k, ch in enumerate(CHANNELS):
        m = known[:, k]
        y, s = pos[m, k], score[m, k]
        _, ap = auroc_ap(y, s)
        r = {"n": int(m.sum()), "pos": int(y.sum()), "ap": ap}
        if thr is not None and thr.get(ch) is not None and y.any():
            r["miss_at_dev_thr"] = float((s[y] < thr[ch]).mean())
            r["fpr_at_dev_thr"] = float((s[~y] >= thr[ch]).mean()) if (~y).any() else None
        res[ch] = r
    return res


def pair_ordering(meta, rows, lgs):
    """Pairs = rows sharing (root_uid, t0) at a branch point whose channel outcome differs."""
    pos, known, _, _ = chunk_labels(meta, rows, lgs[next(iter(lgs))][rows])
    scores = {name: chunk_labels(meta, rows, lg[rows])[2] for name, lg in lgs.items()}
    groups = collections.defaultdict(list)
    for j, r in enumerate(rows):
        groups[(str(meta["root_uid"][r]), int(meta["t0"][r]))].append(j)
    res = {}
    for k, ch in enumerate(CHANNELS):
        hits = {name: [] for name in lgs}
        n_pairs, n_groups = 0, set()
        for key, js in groups.items():
            js = [j for j in js if known[j, k]]
            for a in range(len(js)):
                for b in range(a + 1, len(js)):
                    ja, jb = js[a], js[b]
                    if pos[ja, k] == pos[jb, k]:
                        continue
                    u, s = (ja, jb) if pos[ja, k] else (jb, ja)
                    n_pairs += 1
                    n_groups.add(key[0])
                    for name, sc in scores.items():
                        hits[name].append(float(sc[u, k] > sc[s, k]))
        res[ch] = {"pairs": n_pairs, "roots": len(n_groups),
                   **{f"acc_{name}": (float(np.mean(h)) if h else None) for name, h in hits.items()}}
    return res


def seed_stability(meta, lg_gen, thr):
    """Seed rows vs the tf row generated from the same real chunk."""
    tf_of = {int(meta["src_row"][r]): r for r in np.flatnonzero(meta["mode"] == 0)}
    by_src = collections.defaultdict(list)
    for r in np.flatnonzero(meta["mode"] == 2):
        by_src[int(meta["src_row"][r])].append(r)
    res = {}
    for k, ch in enumerate(CHANNELS):
        stds, vanish, flips, n_pos = [], 0, 0, 0
        for src, rs in by_src.items():
            if src not in tf_of:
                continue
            rows = np.array([tf_of[src]] + rs)
            pos, known, score, _ = chunk_labels(meta, rows, lg_gen[rows])
            if not known[0, k]:
                continue
            sc = score[:, k]
            stds.append(float(np.std(1 / (1 + np.exp(-sc)))))
            above = sc >= thr[ch]
            flips += int(above.any() and not above.all())
            if pos[0, k]:
                n_pos += 1
                vanish += int((~above).any())
        res[ch] = {"chunks": len(stds), "mean_prob_std": float(np.mean(stds)) if stds else None,
                   "decision_flip_rate": flips / len(stds) if stds else None,
                   "unsafe_chunks": n_pos, "unsafe_missed_by_some_seed": vanish / n_pos if n_pos else None}
    return res


def fidelity(gen_mm, real_mm, rows, bs=512):
    cos = {"agent": [], "wrist": []}
    for s in range(0, len(rows), bs):
        rr = rows[s:s + bs]
        g = np.asarray(gen_mm[rr], np.float32)[:, :, 2:]
        r = np.asarray(real_mm[rr], np.float32)[:, :, 2:]
        for v, sl in (("agent", slice(0, 16)), ("wrist", slice(16, 32))):
            a, b = g[..., sl, :].reshape(len(rr), -1), r[..., sl, :].reshape(len(rr), -1)
            cos[v].append((a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12))
    return {v: np.concatenate(c) for v, c in cos.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", default=str(OUT_ROOT / "pairs" / "pilot_v1"))
    ap.add_argument("--ckpt", default=str(OUT_ROOT / "checkpoints"))
    ap.add_argument("--out", default=str(OUT_ROOT / "reports" / "step5_pilot.json"))
    args = ap.parse_args()
    pdir, cdir = Path(args.pairs), Path(args.ckpt)
    with np.load(pdir / "meta.npz") as m:
        meta = {k: m[k] for k in m.files}
    gen_mm = np.load(pdir / "gen_z.npy", mmap_mode="r")
    real_mm = np.load(pdir / "real_z.npy", mmap_mode="r")
    thr = {v: json.loads((cdir / f"cost_real_{v}" / "final.json").read_text())["thresholds_from_dev_r90"]
           for v in ("B0", "B1", "B2")}

    lg = {}
    for v in ("B0", "B1", "B2"):
        model = load_model(cdir / f"cost_real_{v}" / "best.pt")
        lg[f"{v}_real"] = logits(model, real_mm, meta)
        if v != "B0":
            lg[f"{v}_gen"] = logits(model, gen_mm, meta)
        del model
        torch.cuda.empty_cache()

    mode, depth = meta["mode"], meta["depth"]
    subsets = {"tf_all": np.flatnonzero(mode == 0), "tf_uniform": np.flatnonzero((mode == 0) & meta["in_uniform"])}
    for name, (lo, hi) in DEPTH_BUCKETS.items():
        subsets[f"ar_{name}"] = np.flatnonzero((mode == 1) & (depth >= lo) & (depth <= hi))
    rep = {"rows": {k: int(len(v)) for k, v in subsets.items()}, "ap": {}}
    for sname, rows in subsets.items():
        rep["ap"][sname] = {name: chunk_ap(meta, rows, l, thr[name.split("_")[0]]) for name, l in lg.items()}

    pair_rows = np.flatnonzero((mode == 0) & meta["is_pair"])
    rep["pair_ordering_tf"] = pair_ordering(meta, pair_rows, {k: lg[k] for k in ("B0_real", "B1_real", "B2_real", "B2_gen")})
    rep["seed_stability_B2_gen"] = seed_stability(meta, lg["B2_gen"], thr["B2"])

    fid_rows = np.flatnonzero(mode != 2)
    cos = fidelity(gen_mm, real_mm, fid_rows)
    quad = meta["quadrant"][fid_rows]
    is_ev = meta["is_event"][fid_rows]
    fid = {}
    for sname, sel in [("tf", mode[fid_rows] == 0)] + [
            (f"ar_{n}", (mode[fid_rows] == 1) & (depth[fid_rows] >= lo) & (depth[fid_rows] <= hi))
            for n, (lo, hi) in DEPTH_BUCKETS.items()]:
        fid[sname] = {v: float(np.median(c[sel])) if sel.any() else None for v, c in cos.items()}
        fid[sname]["event_wrist"] = float(np.median(cos["wrist"][sel & is_ev])) if (sel & is_ev).any() else None
        fid[sname]["Q4_wrist"] = float(np.median(cos["wrist"][sel & (quad == 3)])) if (sel & (quad == 3)).any() else None
    rep["future_cosine_median"] = fid
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1))

    def f(x):
        return "  -  " if x is None else f"{x:.3f}"
    print("rows", rep["rows"])
    for sname, d in rep["ap"].items():
        print(f"== {sname}")
        for name, r in d.items():
            print(f"  {name:8s} " + " ".join(f"{ch} AP {f(r[ch]['ap'])} miss {f(r[ch].get('miss_at_dev_thr'))} "
                                             f"(pos {r[ch]['pos']})" for ch in CHANNELS))
    print("pair ordering", json.dumps(rep["pair_ordering_tf"]))
    print("seed stability", json.dumps(rep["seed_stability_B2_gen"]))
    print("future cosine", json.dumps(rep["future_cosine_median"]))


if __name__ == "__main__":
    main()
