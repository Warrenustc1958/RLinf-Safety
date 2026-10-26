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

"""Step 7 v0: safety-underestimation head h on a frozen Step 6 cost model.

The cost model (``--cost``) is frozen. On ``pairs/h_train_v1`` (not used to fit or
select the cost model) each row is scored on its generated latent, and per channel
k with a fully known chunk label:

  p[t,k]          = sigmoid(frozen_cost_logit[t,k])
  error_target[k] = max_valid_t max(0, cost[t,k] - p[t,k])

A small MLP predicts error_target (masked Huber). Predictors:
  simple   AR depth bucket + task + suite + cost chunk scores
  h        frozen pooled trunk features (mean/max over valid steps) + cost chunk scores
  h_depth  h + depth bucket
All MLPs are trained on an 80/20 group split of h_train (early stop on the 20%),
``--seeds`` times. Evaluation is on ``pairs/pilot_v1`` (TF + AR rows, generated):

  ranking  per channel AP of "missed danger" (positive chunk scored below the
           R90 threshold fixed on h_train TF rows) and Spearman on positives
  gating   Y_any = any channel positive. cost_only accepts the lowest-risk rows by
           r = max_k s_k / thr_k; cost+X first applies the R90 cost gate and then
           rejects the rows with the largest predicted error. Methods are compared
           at the same coverage: accepted unsafe rate, miss rate, with a group
           bootstrap CI of the difference h - simple and h - cost_only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from toolkits.libero_safety.sam_alignment.model import SAMNet
from toolkits.libero_safety.sam_alignment.schema import CHANNELS, CHUNK, OUT_ROOT
from toolkits.libero_safety.sam_alignment.train import auroc_ap, chunk_labels, thr_at_recall

DEPTH_EDGES = [(0, 0), (1, 2), (3, 4), (5, 8), (9, 16), (17, 99)]  # bucket 0 = TF
K = len(CHANNELS)


def load_meta(path):
    with np.load(path) as m:
        return {k: m[k] for k in m.files}


def depth_bucket(meta):
    d = np.where(meta["mode"] == 0, 0, meta["depth"].astype(int))
    b = np.zeros(len(d), np.int64)
    for i, (lo, hi) in enumerate(DEPTH_EDGES):
        b[(d >= lo) & (d <= hi)] = i
    return b


@torch.no_grad()
def frozen_outputs(model, z_mm, meta, rows, bs=256):
    """Frozen cost logits [N, 8, K] and per-step trunk features [N, 8, dim]."""
    model.eval()
    lgs, hs = [], []
    for s in range(0, len(rows), bs):
        r = rows[s:s + bs]
        z = torch.from_numpy(np.asarray(z_mm[np.sort(r)], np.float32)[np.argsort(np.argsort(r))]).cuda()
        a = torch.from_numpy(meta["action"][r]).cuda()
        t = torch.from_numpy(meta["task_id"][r].astype(np.int64)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = [model.act((a - model.act_mu) / model.act_sd) + model.step_emb, model.task_emb(t)[:, None],
                      model.visual_tokens(z)]
            h = model.norm(model.encoder(torch.cat(tokens, 1)))[:, :CHUNK]
            lg = model.cost_head(h)
        lgs.append(lg.float().cpu().numpy())
        hs.append(h.float().cpu().numpy())
    return np.concatenate(lgs), np.concatenate(hs)


def build(meta, rows, logits, feats, thr):
    """Inputs per predictor, error targets, chunk labels."""
    pos, known, score, _ = chunk_labels(meta, rows, logits)          # [N, K]
    vc, c, vs = meta["valid_channel"][rows], meta["cost"][rows], meta["valid_step"][rows]
    p = 1 / (1 + np.exp(-logits))
    err = np.where(vc, np.maximum(0.0, c.astype(np.float32) - p), 0.0).max(1)  # [N, K]
    s = 1 / (1 + np.exp(-np.where(np.isfinite(score), score, -30.0)))      # chunk score (prob)
    s_mean = (np.where(vc, p, 0).sum(1) / np.maximum(vc.sum(1), 1))
    m = vs[:, :, None].astype(np.float32)
    f_mean = (feats * m).sum(1) / np.maximum(m.sum(1), 1)
    f_max = np.where(vs[:, :, None], feats, -1e4).max(1)
    db = depth_bucket({k: meta[k][rows] for k in ("mode", "depth")})
    onehot = lambda v, n: np.eye(n, dtype=np.float32)[v]
    scores = np.concatenate([s, s_mean, s / np.array([thr[ch] for ch in CHANNELS])[None]], 1).astype(np.float32)
    x = {"simple": np.concatenate([onehot(db, len(DEPTH_EDGES)), onehot(meta["task_id"][rows].astype(int), 15),
                                   onehot(meta["suite"][rows].astype(int), 4), scores], 1),
         "h": np.concatenate([f_mean, f_max, scores], 1)}
    x["h_depth"] = np.concatenate([x["h"], onehot(db, len(DEPTH_EDGES))], 1)
    return {"x": {k: v.astype(np.float32) for k, v in x.items()}, "err": err.astype(np.float32), "known": known,
            "app": vc.any(1),
            "pos": pos, "s": s, "db": db, "group": meta["group"][rows]}


class ErrMLP(nn.Module):
    def __init__(self, d_in, hid=128, p=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hid), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(hid, hid), nn.GELU(), nn.Dropout(p), nn.Linear(hid, K))

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def fit(x_tr, y_tr, m_tr, x_va, y_va, m_va, seed, epochs=200, lr=1e-3, wd=1e-3, bs=256, patience=20):
    torch.manual_seed(seed)
    net = ErrMLP(x_tr.shape[1]).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    T = lambda a: torch.from_numpy(a).cuda()
    xt, yt, mt, xv, yv, mv = T(x_tr), T(y_tr), T(m_tr.astype(np.float32)), T(x_va), T(y_va), T(m_va.astype(np.float32))

    def loss_of(pred, y, m):
        return (F.smooth_l1_loss(pred, y, beta=0.1, reduction="none") * m).sum() / m.sum().clamp_min(1)

    g = torch.Generator(device="cuda").manual_seed(seed)
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xt), device="cuda", generator=g)
        for i in range(0, len(xt), bs):
            j = perm[i:i + bs]
            loss = loss_of(net(xt[j]), yt[j], mt[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_of(net(xv), yv, mv).item()
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, best, ep


def spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def ranking_report(d, pred, thr, rows_sel):
    out = {}
    for k, ch in enumerate(CHANNELS):
        m = d["known"][rows_sel, k]
        r = rows_sel[m]
        miss = d["pos"][r, k] & (d["s"][r, k] < thr[ch])
        a, ap = auroc_ap(miss, pred[r, k])
        posr = r[d["pos"][r, k]]
        out[ch] = {"n": int(len(r)), "pos": int(d["pos"][r, k].sum()), "missed": int(miss.sum()),
                   "missed_auroc": a, "missed_ap": ap, "missed_base": float(miss.mean()) if len(r) else None,
                   "spearman_pos": spearman(pred[posr, k], d["err"][posr, k])}
    return out


def gate_curves(d, preds, thr, rows_sel, rel_cov=(1.0, 0.95, 0.9, 0.85, 0.8, 0.7), boot=500, seed=0):
    """Accepted unsafe rate / miss rate at matched coverage, with group bootstrap of differences."""
    # a channel never valid in the chunk is not applicable (e.g. c_grasp in non-grasp tasks); a row is
    # usable when every applicable channel has a fully known label
    app_all = d["app"][rows_sel]
    full = ((d["known"][rows_sel] | ~app_all).all(1)) & app_all.any(1)
    r = rows_sel[full]
    app = d["app"][r]
    y = d["pos"][r].any(1)
    tk = np.array([thr[ch] for ch in CHANNELS])[None]
    risk = (d["s"][r] / tk).max(1)
    gate = (d["s"][r] <= tk).all(1)
    n, c0 = len(r), float(gate.mean())
    groups = d["group"][r]

    def accept_cost(cov):
        m = int(round(cov * n))
        acc = np.zeros(n, bool)
        acc[np.argsort(risk, kind="mergesort")[:m]] = True
        return acc

    def accept_err(e, cov):
        m = int(round(cov * n))
        acc = np.zeros(n, bool)
        idx = np.flatnonzero(gate)
        acc[idx[np.argsort(e[idx], kind="mergesort")[:m]]] = True
        return acc

    methods = {"cost_only": lambda cov: accept_cost(cov)}
    for name, p in preds.items():
        e = np.where(app, p[r], 0.0).max(1)
        methods[f"cost+{name}"] = (lambda e: lambda cov: accept_err(e, cov))(e)
    rng = np.random.default_rng(seed)
    methods["cost+random"] = (lambda e: lambda cov: accept_err(e, cov))(rng.random(n))
    ug = np.unique(groups)
    idx_of = {u: np.flatnonzero(groups == u) for u in ug}
    takes = [np.concatenate([idx_of[u] for u in rng.choice(ug, len(ug))]) for _ in range(boot)]
    res = {"n": n, "unsafe": int(y.sum()), "base_rate": float(y.mean()), "cost_gate_coverage": c0, "points": []}
    for rc in rel_cov:
        cov = c0 * rc
        pt = {"rel": rc, "coverage": cov}
        accs = {mname: f(cov) for mname, f in methods.items()}
        for mname, acc in accs.items():
            au = int((acc & y).sum())
            pt[mname] = {"accepted_unsafe": au, "accepted_unsafe_rate": au / max(acc.sum(), 1),
                         "miss_rate": au / max(y.sum(), 1)}
        for a_name, b_name in (("cost+h", "cost+simple"), ("cost+h", "cost_only"), ("cost+simple", "cost_only"),
                               ("cost+h_depth", "cost+simple")):
            if a_name not in accs or b_name not in accs:
                continue
            da = (accs[a_name] & y).astype(float) - (accs[b_name] & y).astype(float)
            diffs = [da[t].sum() / max(y[t].sum(), 1) for t in takes]
            pt[f"dmiss[{a_name}-{b_name}]"] = {"mean": float(da.sum() / max(y.sum(), 1)),
                                                "ci95": [float(np.percentile(diffs, 2.5)),
                                                         float(np.percentile(diffs, 97.5))]}
        res["points"].append(pt)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cost", default=str(OUT_ROOT / "checkpoints" / "gen_S2_gen" / "best.pt"))
    ap.add_argument("--h_train", default=str(OUT_ROOT / "pairs" / "h_train_v1"))
    ap.add_argument("--pilot", default=str(OUT_ROOT / "pairs" / "pilot_v1"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--holdout_frac", type=float, default=0.0,
                    help="group-disjoint share of --h_train kept out of thresholds / h fitting, used as eval set")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    t_start = time.time()

    def log(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(out / "log.txt", "a") as f:
            f.write(line + "\n")

    ck = torch.load(args.cost, map_location="cpu", weights_only=False)
    model = SAMNet(**ck["config"]).cuda()
    model.load_state_dict(ck["model_state_dict"])
    for p in model.parameters():
        p.requires_grad_(False)

    hm = load_meta(Path(args.h_train) / "meta.npz")
    pm = load_meta(Path(args.pilot) / "meta.npz")
    h_rows = np.flatnonzero(np.isin(hm["mode"], (0, 1)))
    p_rows = np.flatnonzero(np.isin(pm["mode"], (0, 1)))
    lg_h, f_h = frozen_outputs(model, np.load(Path(args.h_train) / "gen_z.npy", mmap_mode="r"), hm, h_rows)
    lg_p, f_p = frozen_outputs(model, np.load(Path(args.pilot) / "gen_z.npy", mmap_mode="r"), pm, p_rows)
    log(f"frozen outputs: h_train {len(h_rows)} rows, pilot {len(p_rows)} rows")

    # group-disjoint holdout of h_train (eval only: never used for thresholds or fitting)
    rng = np.random.default_rng(0)
    hg = hm["group"][h_rows]
    ug_all = np.unique(hg)
    ho_groups = set(rng.choice(ug_all, int(round(args.holdout_frac * len(ug_all))), replace=False)) \
        if args.holdout_frac > 0 else set()
    ho = np.array([g in ho_groups for g in hg])

    # R90 thresholds on the fitting part of h_train TF rows (generated), never on the pilot or holdout.
    # The recall threshold depends only on positive scores.
    tfu = np.flatnonzero((hm["mode"][h_rows] == 0) & ~ho)
    pos, known, score, _ = chunk_labels(hm, h_rows[tfu], lg_h[tfu])
    thr = {}
    for k, ch in enumerate(CHANNELS):
        t = thr_at_recall(pos[known[:, k], k], score[known[:, k], k])
        thr[ch] = float(1 / (1 + np.exp(-t)))
    log(f"R90 thresholds (h_train TF, prob) {thr}")

    dh = build(hm, h_rows, lg_h, f_h, thr)
    dp = build(pm, p_rows, lg_p, f_p, thr)
    ug = np.unique(dh["group"][~ho])
    va_groups = set(rng.choice(ug, int(round(args.val_frac * len(ug))), replace=False))
    va = np.array([g in va_groups for g in dh["group"]]) & ~ho
    tr = ~va & ~ho
    log(f"h_train split: fit {int(tr.sum())} / val {int(va.sum())} / holdout {int(ho.sum())} rows, "
        f"groups {len(ug_all)} (holdout {len(ho_groups)}); error>0 rows per channel (fit+val) "
        f"{[int(((dh['err'][~ho, k] > 0) & dh['known'][~ho, k]).sum()) for k in range(K)]}")

    def buckets(mode, depth, base, prefix):
        out = {f"{prefix}_all": base, f"{prefix}_tf": base[mode[base] == 0], f"{prefix}_ar": base[mode[base] == 1]}
        for name, (lo, hi) in (("d1_4", (1, 4)), ("d5_8", (5, 8)), ("d9_16", (9, 16)), ("d17_32", (17, 32))):
            r = base[(mode[base] == 1) & (depth[base] >= lo) & (depth[base] <= hi)]
            if len(r):
                out[f"{prefix}_ar_{name}"] = r
        return out

    # eval sets: (built dict, row indices into it)
    sel = {k: ("p", r) for k, r in buckets(pm["mode"][p_rows], pm["depth"][p_rows], np.arange(len(p_rows)),
                                          "pilot").items()}
    if ho.any():
        sel.update({k: ("h", r) for k, r in buckets(hm["mode"][h_rows], hm["depth"][h_rows], np.flatnonzero(ho),
                                                   "hold").items()})
    report = {"cost": args.cost, "thr_r90_h_train_tf": thr, "holdout_groups": len(ho_groups), "seeds": {}}
    main_set = "hold_all" if ho.any() else "pilot_all"
    for seed in range(args.seeds):
        preds, fitinfo = {"p": {}, "h": {}}, {}
        for name in ("simple", "h", "h_depth"):
            x = dh["x"][name]
            net, vloss, ep = fit(x[tr], dh["err"][tr], dh["known"][tr], x[va], dh["err"][va], dh["known"][va],
                                 seed=seed)
            with torch.no_grad():
                preds["p"][name] = net(torch.from_numpy(dp["x"][name]).cuda()).cpu().numpy()
                preds["h"][name] = net(torch.from_numpy(x).cuda()).cpu().numpy()
            fitinfo[name] = {"val_huber": vloss, "epochs": ep + 1}
        rs = {"fit": fitinfo, "ranking": {}, "gating": {}}
        for sname, (src, r) in sel.items():
            d, pr = (dp, preds["p"]) if src == "p" else (dh, preds["h"])
            rs["ranking"][sname] = {name: ranking_report(d, p, thr, r) for name, p in pr.items()}
            rs["gating"][sname] = gate_curves(d, pr, thr, r, seed=seed)
        report["seeds"][seed] = rs
        g = rs["gating"][main_set]["points"]
        log(f"seed {seed} fit {fitinfo} | {main_set} miss@rel " + " ".join(
            f"{p['rel']}: cost {p['cost_only']['miss_rate']:.3f} simple {p['cost+simple']['miss_rate']:.3f} "
            f"h {p['cost+h']['miss_rate']:.3f} hd {p['cost+h_depth']['miss_rate']:.3f}" for p in g))
    report["hours"] = (time.time() - t_start) / 3600
    (out / "final.json").write_text(json.dumps(report, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    log("done")


if __name__ == "__main__":
    main()
