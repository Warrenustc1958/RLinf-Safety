"""Compare proprio state models for imagined rollouts: open-loop error at 8 / 32 / 64 steps.

Models (all predict d_t = s_{t+1} - s_t from history, rolled out open loop from real s_t0):
  lin1   [a_t, grip_t, d_{t-1}, 1]                      (current keyframe-builder model)
  arxL   [a_t..a_{t-L+1}, d_{t-1}..d_{t-L}, grip_t, 1]  least squares, L = 2 / 4 / 8
  mlp    same features as arx4, 2x256 MLP
Writes the best model to --out (npz with W or MLP weights + feature spec) and a report.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from toolkits.libero_safety.sam_alignment.schema import OUT_ROOT
from toolkits.libero_safety.sam_alignment.scgrpo.build_keyframes import ROLES, episode_states


def load(e):
    st = episode_states(e)
    with np.load(e["labels"]) as z:
        act = z["action_env"].astype(np.float32)
    return st.astype(np.float64), act[: len(st) - 1].astype(np.float64)


def feats(acts, deltas, grip, L):
    """acts [L,7] newest first, deltas [L,8] newest first, grip [2]."""
    return np.concatenate([acts[:L].ravel(), deltas[:L].ravel(), grip, [1.0]])


def build_xy(trans, L):
    X, Y = [], []
    for st, act in trans:
        d = np.diff(st, axis=0)
        n = len(d)
        for t in range(L, n):
            X.append(feats(act[t - L + 1:t + 1][::-1], d[t - L:t][::-1], st[t, 6:8], L))
            Y.append(d[t])
    return np.array(X), np.array(Y)


def rollout(pred_fn, st, act, t0, H, L):
    d_hist = list(np.diff(st[: t0 + 1], axis=0)[-L:][::-1])
    a_hist = list(act[t0 - L + 1:t0][::-1]) if L > 1 else []
    s = st[t0].copy()
    out = []
    for h in range(H):
        a_hist = [act[t0 + h]] + a_hist
        x = feats(np.array(a_hist[:L]), np.array(d_hist[:L]), s[6:8], L)
        d = pred_fn(x[None])[0]
        s = s + d
        d_hist = [d] + d_hist
        out.append(s.copy())
    return np.array(out)


def evaluate(pred_fn, trans, L, H=64):
    errs = []
    for st, act in trans:
        for t0 in range(max(8, L + 1), len(st) - H - 1, 32):
            p = rollout(pred_fn, st, act, t0, H, L)
            errs.append(np.abs(p - st[t0 + 1:t0 + H + 1]))
    e = np.stack(errs)
    return {f"pos_{h}": float(e[:, h - 1, :3].mean()) for h in (8, 32, 64)} | \
           {f"rot_{h}": float(e[:, h - 1, 3:6].mean()) for h in (8, 32, 64)} | \
           {f"grip_{h}": float(e[:, h - 1, 6:8].mean()) for h in (8, 64)}


class MLP(torch.nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d_in, 256), torch.nn.GELU(), torch.nn.Linear(256, 256),
                                       torch.nn.GELU(), torch.nn.Linear(256, 8))

    def forward(self, x):
        return self.net(x)


def fit_mlp(X, Y, epochs=30):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    ymu, ysd = Y.mean(0), Y.std(0) + 1e-8
    net = MLP(X.shape[1]).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32).cuda()
    Yt = torch.tensor((Y - ymu) / ysd, dtype=torch.float32).cuda()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), device="cuda")
        for i in range(0, len(Xt), 4096):
            j = perm[i:i + 4096]
            loss = torch.nn.functional.smooth_l1_loss(net(Xt[j]), Yt[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()

    def pred(x):
        with torch.no_grad():
            y = net(torch.tensor((x - mu) / sd, dtype=torch.float32).cuda()).cpu().numpy()
        return y * ysd + ymu
    state = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
    return pred, {"mu": mu, "sd": sd, "ymu": ymu, "ysd": ysd, **{f"w_{k}": v for k, v in state.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT_ROOT / "scgrpo" / "state_model_v1.npz"))
    ap.add_argument("--n_eps", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    reg = json.load(open(OUT_ROOT / "splits" / "split_registry.json"))
    eps = [json.loads(l) for l in open(OUT_ROOT / "labels" / "episodes.jsonl")]
    eps = [e for e in eps if reg["episode_role"].get(e["uid"]) in ROLES and e["label_source"] == "grader_v1"
           and e["dual"]]
    rng = np.random.default_rng(0)
    eps = [eps[i] for i in rng.permutation(len(eps))[: args.n_eps]]
    with Pool(args.workers) as p:
        trans = p.map(load, eps, chunksize=8)
    cut = int(0.85 * len(trans))
    tr, te = trans[:cut], trans[cut:]
    te_eval = te[:300]
    rep, best = {}, None
    for L in (1, 2, 4, 8):
        X, Y = build_xy(tr, L)
        W = np.linalg.lstsq(X, Y, rcond=None)[0]
        r = evaluate(lambda x, W=W: x @ W, te_eval, L)
        rep[f"arx{L}"] = r
        print(f"arx{L}", {k: round(v, 4) for k, v in r.items()}, flush=True)
        if best is None or r["pos_32"] < best[0]["pos_32"]:
            best = (r, {"kind": "arx", "L": L, "W": W})
    X, Y = build_xy(tr, 4)
    pred, w = fit_mlp(X, Y)
    r = evaluate(pred, te_eval, 4)
    rep["mlp4"] = r
    print("mlp4", {k: round(v, 4) for k, v in r.items()}, flush=True)
    if r["pos_32"] < best[0]["pos_32"]:
        best = (r, {"kind": "mlp", "L": 4, **w})
    # zero-motion baseline (hold the keyframe state)
    rep["hold"] = evaluate(lambda x: np.zeros((len(x), 8)), te_eval, 1)
    print("hold", {k: round(v, 4) for k, v in rep["hold"].items()}, flush=True)
    sel = best[1]
    np.savez(args.out, kind=np.array(sel.pop("kind")), L=np.array(sel.pop("L")), **sel)
    rep["selected"] = str(np.load(args.out)["kind"]) + str(int(np.load(args.out)["L"]))
    Path(args.out).with_suffix(".json").write_text(json.dumps(rep, indent=1))
    print("selected", rep["selected"])


if __name__ == "__main__":
    main()
