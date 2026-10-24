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

"""Step 4: train and evaluate the real-domain SAM baselines B0 / B1 / B2.

Loss: masked BCEWithLogits on per-step channel costs (pos_weight = neg/pos per
channel, capped) + 0.1 x masked smooth-L1 on proximity for c_obj / c_robot.
Sampling: initial-state-group balanced, event chunks boosted to ``--event_share``.
Model selection: mean chunk-level AP over channels (>= 20 positive chunks) on the
dev uniform subset. Thresholds at 90 % chunk recall are fixed on dev and only
applied to diag_newtest. Group-bootstrap CIs and the future-swap diagnostic are
reported for the selected checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from toolkits.libero_safety.sam_alignment.model import SAMNet
from toolkits.libero_safety.sam_alignment.schema import CHANNELS, OUT_ROOT


class SAMChunks(Dataset):
    def __init__(self, root, role, load_z=True, only_uniform=False):
        root = Path(root)
        with np.load(root / f"{role}_meta.npz") as m:
            self.meta = {k: m[k] for k in m.files}
        n = len(self.meta["t0"])
        self.rows = np.flatnonzero(self.meta["in_uniform"]) if only_uniform else np.arange(n)
        self.path, self.load_z, self._z = root / f"{role}_z.npy", load_z, None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        j = int(self.rows[i])
        if self.load_z:
            if self._z is None:
                self._z = np.load(self.path, mmap_mode="r")
            z = torch.from_numpy(np.array(self._z[j], dtype=np.float32))
        else:
            z = torch.zeros(1)
        m = self.meta
        return (z, torch.from_numpy(m["action"][j]), int(m["task_id"][j]), torch.from_numpy(m["cost"][j]),
                torch.from_numpy(m["valid_channel"][j]), torch.from_numpy(m["prox"][j]),
                torch.from_numpy(m["prox_valid"][j]), j)


# ---------------------------------------------------------------- metrics
def _rank(s):
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    return (np.bincount(inv, weights=r) / cnt)[inv]


def auroc_ap(y, s):
    pos, neg = int(y.sum()), int((~y).sum())
    if pos == 0 or neg == 0:
        return None, None
    auroc = (_rank(s)[y].sum() - pos * (pos + 1) / 2) / (pos * neg)
    ys = y[np.argsort(-s, kind="mergesort")]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return float(auroc), float(prec[ys].mean())


def thr_at_recall(y, s, recall=0.9):
    sp = np.sort(s[y])[::-1]
    return float(sp[max(0, int(math.ceil(recall * len(sp))) - 1)]) if len(sp) else None


def chunk_labels(meta, rows, logits):
    """Per channel: positive (any valid violation), label known, max-logit score, near-miss."""
    vc, c, vs = meta["valid_channel"][rows], meta["cost"][rows], meta["valid_step"][rows]
    pos = (c & vc).any(1)
    known = pos | (((vc | ~vs[:, :, None]).all(1)) & vs.any(1)[:, None])
    score = np.where(vc, logits, -np.inf).max(1)
    near = (meta["near_miss"][rows] & vc).any(1)
    return pos, known, score, near


def evaluate(meta, rows, logits, thresholds=None, bootstrap=0, seed=0):
    pos, known, score, near = chunk_labels(meta, rows, logits)
    vc, c = meta["valid_channel"][rows], meta["cost"][rows]
    groups = meta["group"][rows]
    out = {}
    for k, ch in enumerate(CHANNELS):
        r = {}
        st_auroc, st_ap = auroc_ap(c[..., k][vc[..., k]], logits[..., k][vc[..., k]])
        r["step"] = {"auroc": st_auroc, "ap": st_ap, "pos": int((c[..., k] & vc[..., k]).sum()),
                     "valid": int(vc[..., k].sum())}
        m = known[:, k]
        y, s, nm, g = pos[m, k], score[m, k], near[m, k], groups[m]
        a, ap = auroc_ap(y, s)
        thr = thr_at_recall(y, s)
        r["chunk"] = {"n": int(m.sum()), "pos": int(y.sum()), "base_rate": float(y.mean()) if len(y) else None,
                      "groups_with_pos": int(len(set(g[y]))), "auroc": a, "ap": ap, "thr_r90": thr,
                      "fpr_at_r90": float((s[~y] >= thr).mean()) if thr is not None and (~y).any() else None,
                      "nearmiss_fpr_at_r90": float((s[~y & nm] >= thr).mean())
                      if thr is not None and (~y & nm).any() else None}
        if thresholds and thresholds.get(ch) is not None:
            t = thresholds[ch]
            r["at_dev_thr"] = {"thr": t, "recall": float((s[y] >= t).mean()) if y.any() else None,
                               "fpr": float((s[~y] >= t).mean()) if (~y).any() else None,
                               "nearmiss_fpr": float((s[~y & nm] >= t).mean()) if (~y & nm).any() else None}
        if bootstrap and y.any() and (~y).any():
            rng = np.random.default_rng(seed)
            ug = np.unique(g)
            idx_of = {u: np.flatnonzero(g == u) for u in ug}
            aps = []
            for _ in range(bootstrap):
                take = np.concatenate([idx_of[u] for u in rng.choice(ug, len(ug))])
                v = auroc_ap(y[take], s[take])[1]
                if v is not None:
                    aps.append(v)
            r["chunk"]["ap_ci95"] = [float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5))]
        out[ch] = r
    return out


def selection_score(res, min_pos=20):
    aps = [r["chunk"]["ap"] for r in res.values() if r["chunk"]["pos"] >= min_pos and r["chunk"]["ap"] is not None]
    return float(np.mean(aps)) if aps else 0.0


@torch.no_grad()
def predict(model, ds, bs, workers, swap_future=False):
    model.eval()
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    outs = []
    for z, a, t, *_ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit, _ = model(z.cuda(non_blocking=True), a.cuda(non_blocking=True), t.cuda(non_blocking=True),
                             swap_future=swap_future)
        outs.append(logit.float().cpu())
    model.train()
    return torch.cat(outs).numpy()


# ---------------------------------------------------------------- training
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default=str(OUT_ROOT / "cache" / "real_v1"))
    ap.add_argument("--variant", choices=("B0", "B1", "B2"), required=True)
    ap.add_argument("--views", type=int, choices=(1, 2), default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--pos_weight_cap", type=float, default=20.0)
    ap.add_argument("--aux_weight", type=float, default=0.1)
    ap.add_argument("--event_share", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=0, help="smoke test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    def log(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(out / "log.txt", "a") as f:
            f.write(line + "\n")

    cache = Path(args.cache)
    stats = json.loads((cache / "latent_stats.json").read_text())
    load_z = args.variant != "B0"
    train = SAMChunks(cache, "cost_train", load_z)
    dev_u = SAMChunks(cache, "dev", load_z, only_uniform=True)
    dev_all = SAMChunks(cache, "dev", load_z)
    diag = SAMChunks(cache, "diag_newtest", load_z)
    num_tasks = int(max(train.meta["task_id"].max(), dev_all.meta["task_id"].max(), diag.meta["task_id"].max())) + 1
    num_tasks = max(num_tasks, 15)

    model = SAMNet(args.variant, args.views, num_tasks, latent_mean=stats["latent_mean"],
                   latent_std=stats["latent_std"], action_mean=stats["action_mean"],
                   action_std=stats["action_std"]).cuda()
    n_params = sum(p.numel() for p in model.parameters())

    tm = train.meta
    pos = (tm["cost"] & tm["valid_channel"]).sum((0, 1)).astype(float)
    neg = (~tm["cost"] & tm["valid_channel"]).sum((0, 1)).astype(float)
    pw = np.minimum(neg / np.maximum(pos, 1), args.pos_weight_cap)
    pos_weight = torch.tensor(pw, dtype=torch.float32).cuda()

    _, ginv, gcnt = np.unique(tm["group"], return_inverse=True, return_counts=True)
    w = 1.0 / gcnt[ginv]
    ev = tm["is_event"]
    if ev.any() and (~ev).any():
        share = w[ev].sum() / w.sum()
        if share < args.event_share:
            w[ev] *= args.event_share * w[~ev].sum() / ((1 - args.event_share) * w[ev].sum())
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(train), replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(train, batch_size=args.bs, sampler=sampler, num_workers=args.workers, pin_memory=True,
                        drop_last=True, persistent_workers=True, prefetch_factor=4)
    total = args.epochs * len(loader)
    if args.max_steps:
        total = min(total, args.max_steps)
    warmup = max(1, int(args.warmup_frac * total))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    log(f"variant={args.variant} views={args.views} params={n_params} train={len(train)} dev_uniform={len(dev_u)} "
        f"dev_all={len(dev_all)} diag={len(diag)} pos_steps={pos.astype(int).tolist()} pos_weight={pw.round(2).tolist()} "
        f"steps/epoch={len(loader)}")

    step, best, bad, t0 = 0, -1.0, 0, time.time()
    for epoch in range(args.epochs):
        run = np.zeros(3)
        for z, a, t, c, vc, px, pv, _ in loader:
            z, a, t = z.cuda(non_blocking=True), a.cuda(non_blocking=True), t.cuda(non_blocking=True)
            c, vc = c.cuda(non_blocking=True).float(), vc.cuda(non_blocking=True).float()
            px, pv = px.cuda(non_blocking=True), pv.cuda(non_blocking=True).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit, prox = model(z, a, t)
            bce = F.binary_cross_entropy_with_logits(logit.float(), c, pos_weight=pos_weight, reduction="none")
            l_event = (bce * vc).sum() / vc.sum().clamp_min(1)
            hub = F.smooth_l1_loss(torch.sigmoid(prox.float()), px, beta=0.1, reduction="none")
            l_aux = (hub * pv).sum() / pv.sum().clamp_min(1)
            loss = l_event + args.aux_weight * l_aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run += [loss.item(), l_event.item(), l_aux.item()]
            if step % 200 == 0:
                log(f"ep {epoch} step {step}/{total} loss {run[0] / 200:.4f} event {run[1] / 200:.4f} "
                    f"aux {run[2] / 200:.4f} lr {sched.get_last_lr()[0]:.2e} {step / (time.time() - t0):.1f} it/s "
                    f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G")
                run[:] = 0
            if step >= total:
                break
        res = evaluate(dev_u.meta, dev_u.rows, predict(model, dev_u, 512, args.workers))
        score = selection_score(res)
        with open(out / "dev_metrics.jsonl", "a") as f:
            f.write(json.dumps({"epoch": epoch, "step": step, "score": score, "dev_uniform": res}) + "\n")
        log(f"ep {epoch} dev-uniform chunk AP " + " ".join(
            f"{ch} {r['chunk']['ap'] if r['chunk']['ap'] is None else round(r['chunk']['ap'], 3)}"
            f"(pos {r['chunk']['pos']})" for ch, r in res.items()) + f" score {score:.4f}")
        torch.save({"model_state_dict": model.state_dict(), "config": model.config(), "epoch": epoch}, out / "last.pt")
        if score > best:
            best, bad = score, 0
            torch.save({"model_state_dict": model.state_dict(), "config": model.config(), "epoch": epoch},
                       out / "best.pt")
            log(f"new best {best:.4f} at epoch {epoch}")
        else:
            bad += 1
            if bad >= args.patience:
                log(f"early stop at epoch {epoch}")
                break
        if step >= total:
            break

    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    lu = predict(model, dev_u, 512, args.workers)
    res_dev = evaluate(dev_u.meta, dev_u.rows, lu, bootstrap=args.bootstrap)
    thr = {ch: res_dev[ch]["chunk"]["thr_r90"] for ch in CHANNELS}
    res_dev_all = evaluate(dev_all.meta, dev_all.rows, predict(model, dev_all, 512, args.workers), thresholds=thr)
    res_diag = evaluate(diag.meta, diag.rows, predict(model, diag, 512, args.workers), thresholds=thr,
                        bootstrap=args.bootstrap)
    final = {"best_epoch": ck["epoch"], "params": n_params, "dev_uniform": res_dev, "dev_all_event_balanced": res_dev_all,
             "diag_newtest": res_diag, "thresholds_from_dev_r90": thr,
             "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "hours": (time.time() - t0) / 3600}
    if args.variant == "B2":
        ls = predict(model, dev_u, 512, args.workers, swap_future=True)
        res_swap = evaluate(dev_u.meta, dev_u.rows, ls)
        vc = dev_u.meta["valid_channel"][dev_u.rows]
        dp = np.abs(1 / (1 + np.exp(-ls)) - 1 / (1 + np.exp(-lu)))
        final["future_swap_dev_uniform"] = {
            ch: {"ap": res_swap[ch]["chunk"]["ap"], "ap_drop": (res_dev[ch]["chunk"]["ap"] or 0) - (res_swap[ch]["chunk"]["ap"] or 0),
                 "mean_abs_dprob": float(dp[..., k][vc[..., k]].mean()) if vc[..., k].any() else None}
            for k, ch in enumerate(CHANNELS)}
    (out / "final.json").write_text(json.dumps(final, indent=1))
    log("final dev-uniform AP " + json.dumps({ch: res_dev[ch]["chunk"]["ap"] for ch in CHANNELS})
        + " diag AP " + json.dumps({ch: res_diag[ch]["chunk"]["ap"] for ch in CHANNELS}))


if __name__ == "__main__":
    main()
