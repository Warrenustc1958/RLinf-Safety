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

"""Step 6: generated-domain SAM cost head, fine-tuned from the real-domain B2.

Training rows come from ``pairs/cost_train_v1`` (TF rows + AR rows up to
``--ar_max_depth``), labels are the real executed outcome. ``--source`` picks the
future latent: ``real`` (budget-matched control), ``gen`` or ``mix`` (each row is
seen both with its real and its generated latent, generated share ``--mix_gen``).

Optional same-state ranking loss (``--rank_weight``): pairs from
``rank_pairs_v1/cost_train_pairs.npz`` (same root_uid + t0, channel k unsafe in one
branch and safe in the other), scored on the generated latent with the chunk
score max_t logit[t, k]; loss softplus(-(S_unsafe - S_safe)). Hard pairs (B0
margin < 1) are oversampled by ``--hard_boost``.

  L = L_event + aux_weight * L_prox + rank_weight * L_rank

Selection (dev = pairs/pilot_v1, generated latent): 0.5 * mean chunk AP on TF
uniform rows + 0.5 * mean chunk AP on AR rows (channels with >= 20 positives).
The final report splits TF / AR depth buckets, real vs generated latent, miss rate
at R90 thresholds fixed on pilot TF-uniform (generated), pair ordering on dev
pairs (all / hard), and real-domain retention on dev-uniform and diag_newtest.
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
from toolkits.libero_safety.sam_alignment.train import chunk_labels, evaluate, selection_score, thr_at_recall

DEPTH_BUCKETS = {"d1-2": (1, 2), "d3-4": (3, 4), "d5-8": (5, 8), "d9-16": (9, 16)}


def load_meta(path):
    with np.load(path) as m:
        return {k: m[k] for k in m.files}


class PairRows(Dataset):
    """Rows of a pairs dir; item i < n uses the real latent, i >= n the generated one."""

    def __init__(self, pdir, rows, meta):
        self.pdir, self.rows, self.meta, self.mm = Path(pdir), rows, meta, None

    def __len__(self):
        return 2 * len(self.rows)

    def __getitem__(self, i):
        if self.mm is None:
            self.mm = (np.load(self.pdir / "real_z.npy", mmap_mode="r"), np.load(self.pdir / "gen_z.npy", mmap_mode="r"))
        src = int(i >= len(self.rows))
        j = int(self.rows[i % len(self.rows)])
        m = self.meta
        z = torch.from_numpy(np.array(self.mm[src][j], dtype=np.float32))
        return (z, torch.from_numpy(m["action"][j]), int(m["task_id"][j]), torch.from_numpy(m["cost"][j]),
                torch.from_numpy(m["valid_channel"][j]), torch.from_numpy(m["prox"][j]),
                torch.from_numpy(m["prox_valid"][j]), j)


@torch.no_grad()
def predict(model, z_mm, meta, rows, bs=256):
    model.eval()
    out = []
    for s in range(0, len(rows), bs):
        r = rows[s:s + bs]
        z = torch.from_numpy(np.asarray(z_mm[np.sort(r)], np.float32)[np.argsort(np.argsort(r))]).cuda()
        a = torch.from_numpy(meta["action"][r]).cuda()
        t = torch.from_numpy(meta["task_id"][r].astype(np.int64)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, _ = model(z, a, t)
        out.append(lg.float().cpu().numpy())
    model.train()
    return np.concatenate(out) if out else np.zeros((0, 8, len(CHANNELS)), np.float32)


def chunk_score_t(logit, vc, k):
    """Differentiable chunk score: max over valid steps of channel k."""
    return logit[..., k].masked_fill(~vc[..., k], -1e4).amax(1)


def map_pairs(pairs, src_row, keep_mode0):
    """Real-cache row pairs -> rows of a pairs dir (TF rows, via src_row)."""
    pos_of = {int(s): n for n, s in enumerate(src_row) if keep_mode0[n]}
    ok = np.array([int(u) in pos_of and int(s) in pos_of for u, s in zip(pairs["unsafe_row"], pairs["safe_row"])])
    u = np.array([pos_of[int(x)] for x in pairs["unsafe_row"][ok]], np.int64)
    s = np.array([pos_of[int(x)] for x in pairs["safe_row"][ok]], np.int64)
    return u, s, pairs["channel"][ok].astype(np.int64), pairs["hard"][ok], float(ok.mean()) if len(ok) else 0.0


def pair_acc(lg, meta, u, s, ch, hard):
    res = {}
    for k, name in enumerate(CHANNELS):
        sel = ch == k
        if not sel.any():
            res[name] = {"pairs": 0}
            continue
        vc = meta["valid_channel"]
        su = np.where(vc[u[sel]][..., k], lg[u[sel]][..., k], -np.inf).max(1)
        ss = np.where(vc[s[sel]][..., k], lg[s[sel]][..., k], -np.inf).max(1)
        win = su > ss
        h = hard[sel]
        res[name] = {"pairs": int(sel.sum()), "acc": float(win.mean()), "hard": int(h.sum()),
                     "acc_hard": float(win[h].mean()) if h.any() else None}
    return res


def pilot_subsets(pm):
    mode, depth = pm["mode"], pm["depth"]
    sub = {"tf_all": np.flatnonzero(mode == 0), "tf_uniform": np.flatnonzero((mode == 0) & pm["in_uniform"]),
           "ar_all": np.flatnonzero(mode == 1)}
    for n, (lo, hi) in DEPTH_BUCKETS.items():
        sub[f"ar_{n}"] = np.flatnonzero((mode == 1) & (depth >= lo) & (depth <= hi))
    return sub


def brief(res):
    return {ch: {"ap": r["chunk"]["ap"], "pos": r["chunk"]["pos"],
                 **({"recall": r["at_dev_thr"]["recall"], "fpr": r["at_dev_thr"]["fpr"]} if "at_dev_thr" in r else {})}
            for ch, r in res.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", default=str(OUT_ROOT / "pairs" / "cost_train_v1"))
    ap.add_argument("--pilot", default=str(OUT_ROOT / "pairs" / "pilot_v1"))
    ap.add_argument("--real_cache", default=str(OUT_ROOT / "cache" / "real_v1"))
    ap.add_argument("--rank_pairs", default=str(OUT_ROOT / "pairs" / "rank_pairs_v1"))
    ap.add_argument("--init", default=str(OUT_ROOT / "checkpoints" / "cost_real_B2" / "best.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", choices=("real", "gen", "mix"), default="mix")
    ap.add_argument("--mix_gen", type=float, default=0.5)
    ap.add_argument("--ar_max_depth", type=int, default=16, help="0 = TF rows only")
    ap.add_argument("--rank_weight", type=float, default=0.0)
    ap.add_argument("--rank_bs", type=int, default=32)
    ap.add_argument("--hard_boost", type=float, default=4.0)
    ap.add_argument("--epochs", type=int, default=10, help="0 = evaluate the init model only")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--pos_weight_cap", type=float, default=20.0)
    ap.add_argument("--aux_weight", type=float, default=0.1)
    ap.add_argument("--event_share", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=0, help="smoke test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude_groups", default="", help="json list of init-state groups dropped from training (CV)")
    ap.add_argument("--no_select", action="store_true",
                    help="no model selection: train --epochs, best.pt = last epoch (dev score only logged)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    def log(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(out / "log.txt", "a") as f:
            f.write(line + "\n")

    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    model = SAMNet(**ck["config"]).cuda()
    model.load_state_dict(ck["model_state_dict"])

    # ------------------------------------------------ training rows
    pdir = Path(args.pairs)
    tm = load_meta(pdir / "meta.npz")
    keep = (tm["mode"] == 0) | ((tm["mode"] == 1) & (tm["depth"] <= args.ar_max_depth))
    if args.exclude_groups:
        keep &= ~np.isin(tm["group"], json.load(open(args.exclude_groups)))
    rows = np.flatnonzero(keep)
    train = PairRows(pdir, rows, tm)
    _, ginv, gcnt = np.unique(tm["group"][rows], return_inverse=True, return_counts=True)
    w = 1.0 / gcnt[ginv]
    ev = tm["is_event"][rows]
    if ev.any() and (~ev).any() and w[ev].sum() / w.sum() < args.event_share:
        w[ev] *= args.event_share * w[~ev].sum() / ((1 - args.event_share) * w[ev].sum())
    w = w / w.sum()
    share_gen = {"real": 0.0, "gen": 1.0, "mix": args.mix_gen}[args.source]
    w2 = np.concatenate([(1 - share_gen) * w, share_gen * w])
    sampler = WeightedRandomSampler(torch.tensor(w2, dtype=torch.double), num_samples=len(rows), replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(train, batch_size=args.bs, sampler=sampler, num_workers=args.workers, pin_memory=True,
                        drop_last=True, persistent_workers=args.workers > 0)

    vc_all, c_all = tm["valid_channel"][rows], tm["cost"][rows]
    pos = (c_all & vc_all).sum((0, 1)).astype(float)
    neg = (~c_all & vc_all).sum((0, 1)).astype(float)
    pw = np.minimum(neg / np.maximum(pos, 1), args.pos_weight_cap)
    pos_weight = torch.tensor(pw, dtype=torch.float32).cuda()

    # ------------------------------------------------ ranking pairs (train)
    rp = load_meta(Path(args.rank_pairs) / "cost_train_pairs.npz")
    ru, rs, rch, rhard, cover = map_pairs(rp, tm["src_row"], tm["mode"] == 0)
    rprob = np.where(rhard, args.hard_boost, 1.0)
    rprob = rprob / rprob.sum()
    rank_mm = np.load(pdir / ("real_z.npy" if args.source == "real" else "gen_z.npy"), mmap_mode="r")

    # ------------------------------------------------ dev (pilot, generated)
    qdir = Path(args.pilot)
    pm = load_meta(qdir / "meta.npz")
    p_gen = np.load(qdir / "gen_z.npy", mmap_mode="r")
    p_real = np.load(qdir / "real_z.npy", mmap_mode="r")
    sub = pilot_subsets(pm)
    dp = load_meta(Path(args.rank_pairs) / "dev_pairs.npz")
    du, ds, dch, dhard, dcover = map_pairs(dp, pm["src_row"], pm["mode"] == 0)

    total = args.epochs * len(loader)
    if args.max_steps:
        total = min(total, args.max_steps)
    warmup = max(1, int(args.warmup_frac * max(total, 1)))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / max(total, 1))))
    log(f"source={args.source} mix_gen={share_gen} ar_max_depth={args.ar_max_depth} rank_weight={args.rank_weight} "
        f"rows={len(rows)} (tf {(tm['mode'][rows] == 0).sum()} ar {(tm['mode'][rows] == 1).sum()}) "
        f"pos_steps={pos.astype(int).tolist()} pos_weight={pw.round(2).tolist()} steps/epoch={len(loader)} "
        f"rank_pairs={len(ru)} (hard {int(rhard.sum())}, cover {cover:.3f}) dev_pairs={len(du)} (cover {dcover:.3f})")

    def dev_score():
        lg_tf = predict(model, p_gen, pm, sub["tf_uniform"])
        lg_ar = predict(model, p_gen, pm, sub["ar_all"])
        r_tf = evaluate(pm, sub["tf_uniform"], lg_tf)
        r_ar = evaluate(pm, sub["ar_all"], lg_ar)
        return 0.5 * selection_score(r_tf) + 0.5 * selection_score(r_ar), r_tf, r_ar

    score0, r_tf, r_ar = dev_score()
    log(f"init dev score {score0:.4f} tf {selection_score(r_tf):.4f} ar {selection_score(r_ar):.4f}")
    best, bad, step, t_start = score0, 0, 0, time.time()
    torch.save({"model_state_dict": model.state_dict(), "config": model.config(), "epoch": -1}, out / "best.pt")

    for epoch in range(args.epochs):
        run = np.zeros(4)
        for z, a, t, c, vc, px, pv, _ in loader:
            z, a, t = z.cuda(non_blocking=True), a.cuda(non_blocking=True), t.cuda(non_blocking=True)
            c, vcb = c.cuda(non_blocking=True).float(), vc.cuda(non_blocking=True)
            vcf = vcb.float()
            px, pv = px.cuda(non_blocking=True), pv.cuda(non_blocking=True).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit, prox = model(z, a, t)
            bce = F.binary_cross_entropy_with_logits(logit.float(), c, pos_weight=pos_weight, reduction="none")
            l_event = (bce * vcf).sum() / vcf.sum().clamp_min(1)
            hub = F.smooth_l1_loss(torch.sigmoid(prox.float()), px, beta=0.1, reduction="none")
            l_aux = (hub * pv).sum() / pv.sum().clamp_min(1)
            loss = l_event + args.aux_weight * l_aux
            l_rank = torch.zeros((), device="cuda")
            if args.rank_weight > 0 and len(ru):
                pick = rng.choice(len(ru), args.rank_bs, p=rprob)
                both = np.concatenate([ru[pick], rs[pick]])
                order = np.argsort(both)
                zz = np.empty((len(both),) + rank_mm.shape[1:], np.float32)
                zz[order] = np.asarray(rank_mm[both[order]], np.float32)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg2, _ = model(torch.from_numpy(zz).cuda(), torch.from_numpy(tm["action"][both]).cuda(),
                                   torch.from_numpy(tm["task_id"][both].astype(np.int64)).cuda())
                vc2 = torch.from_numpy(tm["valid_channel"][both]).cuda()
                k = torch.from_numpy(np.concatenate([rch[pick], rch[pick]])).cuda()
                sc = lg2.float().masked_fill(~vc2, -1e4).amax(1).gather(1, k[:, None])[:, 0]
                l_rank = F.softplus(-(sc[:args.rank_bs] - sc[args.rank_bs:])).mean()
                loss = loss + args.rank_weight * l_rank
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            run += [loss.item(), l_event.item(), l_aux.item(), l_rank.item()]
            if step % 200 == 0:
                log(f"ep {epoch} step {step}/{total} loss {run[0] / 200:.4f} event {run[1] / 200:.4f} "
                    f"aux {run[2] / 200:.4f} rank {run[3] / 200:.4f} lr {sched.get_last_lr()[0]:.2e} "
                    f"{step / (time.time() - t_start):.1f} it/s")
                run[:] = 0
            if step >= total:
                break
        score, r_tf, r_ar = dev_score()
        with open(out / "dev_metrics.jsonl", "a") as f:
            f.write(json.dumps({"epoch": epoch, "step": step, "score": score, "tf_uniform": brief(r_tf),
                                "ar_all": brief(r_ar)}) + "\n")
        log(f"ep {epoch} dev score {score:.4f} | tf " + " ".join(
            f"{ch} {r['chunk']['ap'] and round(r['chunk']['ap'], 3)}" for ch, r in r_tf.items()) + " | ar " + " ".join(
            f"{ch} {r['chunk']['ap'] and round(r['chunk']['ap'], 3)}(pos {r['chunk']['pos']})" for ch, r in r_ar.items()))
        torch.save({"model_state_dict": model.state_dict(), "config": model.config(), "epoch": epoch}, out / "last.pt")
        if args.no_select:
            best = score
            torch.save({"model_state_dict": model.state_dict(), "config": model.config(), "epoch": epoch},
                       out / "best.pt")
        elif score > best:
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

    # ------------------------------------------------ final report
    ckb = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckb["model_state_dict"])
    lg = {"gen": predict(model, p_gen, pm, np.arange(len(pm["t0"]))),
          "real": predict(model, p_real, pm, np.arange(len(pm["t0"])))}
    base = evaluate(pm, sub["tf_uniform"], lg["gen"][sub["tf_uniform"]], bootstrap=args.bootstrap)
    thr = {ch: base[ch]["chunk"]["thr_r90"] for ch in CHANNELS}
    pilot = {}
    for sname, r in sub.items():
        if len(r) == 0:
            continue
        pilot[sname] = {src: brief(evaluate(pm, r, lg[src][r], thresholds=thr)) for src in ("gen", "real")}
    pilot["tf_uniform"]["gen_ap_ci95"] = {ch: base[ch]["chunk"].get("ap_ci95") for ch in CHANNELS}

    cache = Path(args.real_cache)
    real = {}
    for role, only_u in (("dev", True), ("diag_newtest", False)):
        rm = load_meta(cache / f"{role}_meta.npz")
        rr = np.flatnonzero(rm["in_uniform"]) if only_u else np.arange(len(rm["t0"]))
        zmm = np.load(cache / f"{role}_z.npy", mmap_mode="r")
        real[role + ("_uniform" if only_u else "")] = brief(evaluate(rm, rr, predict(model, zmm, rm, rr, bs=512)))
    final = {"best_epoch": ckb["epoch"], "init_dev_score": score0, "best_dev_score": best,
             "thresholds_pilot_tf_uniform_gen_r90": thr, "pilot": pilot,
             "pair_ordering_dev": {src: pair_acc(lg[src], pm, du, ds, dch, dhard) for src in ("gen", "real")},
             "real_domain": real, "hours": (time.time() - t_start) / 3600}
    (out / "final.json").write_text(json.dumps(final, indent=1))
    log("final " + json.dumps({"best_epoch": ckb["epoch"], "score": best,
                               "tf_uniform_gen": {c: v["ap"] for c, v in pilot["tf_uniform"]["gen"].items()},
                               "ar_d9-16_gen": {c: v["ap"] for c, v in pilot["ar_d9-16"]["gen"].items()},
                               "dev_real": {c: v["ap"] for c, v in real["dev_uniform"].items()}}))


if __name__ == "__main__":
    main()
