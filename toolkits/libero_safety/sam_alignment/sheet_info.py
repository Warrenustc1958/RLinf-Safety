"""Metadata + B2 real/gen chunk probabilities for the pilot comparison sheets."""
import json
from pathlib import Path

import numpy as np
import torch

from toolkits.libero_safety.sam_alignment.eval_gen import load_model
from toolkits.libero_safety.sam_alignment.schema import CHANNELS

P = Path("/data_cf/cws_sam/v1/pairs/pilot_v1")
C = Path("/data_cf/cws_sam/v1/checkpoints")
meta = dict(np.load(P / "meta.npz"))
gen = np.load(P / "gen_z.npy", mmap_mode="r")
real = np.load(P / "real_z.npy", mmap_mode="r")
rows = sorted(int(p.stem[3:]) for p in (P / "sheets").glob("row*.png"))
thr = json.loads((C / "cost_real_B2" / "final.json").read_text())["thresholds_from_dev_r90"]
model = load_model(C / "cost_real_B2" / "best.pt")


@torch.no_grad()
def probs(z_mm):
    z = torch.from_numpy(np.asarray(z_mm[rows], np.float32)).cuda()
    a = torch.from_numpy(meta["action"][rows]).cuda()
    t = torch.from_numpy(meta["task_id"][rows].astype(np.int64)).cuda()
    lg, _ = model(z, a, t)
    return lg.float().cpu().numpy()


lr, lg = probs(real), probs(gen)
vc = meta["valid_channel"][rows]
out = []
for j, r in enumerate(rows):
    g, re = np.asarray(gen[r], np.float32)[:, 2:], np.asarray(real[r], np.float32)[:, 2:]
    cos = {}
    for v, sl in (("agent", slice(0, 16)), ("wrist", slice(16, 32))):
        a, b = g[..., sl, :].ravel(), re[..., sl, :].ravel()
        cos[v] = float(a @ b / np.sqrt((a @ a) * (b @ b)))
    info = {"row": r, "mode": ["tf", "ar", "seed"][int(meta["mode"][r])], "depth": int(meta["depth"][r]),
            "t0": int(meta["t0"][r]), "uid": str(meta["uid"][r]), "kind": str(meta["kind"][r]),
            "quadrant": ["Q1", "Q2", "Q3", "Q4"][int(meta["quadrant"][r])], "cos": cos, "ch": {}}
    for k, ch in enumerate(CHANNELS):
        if not vc[j, :, k].any():
            continue
        y = bool((meta["cost"][r][:, k] & vc[j, :, k]).any())
        sr = float(np.where(vc[j, :, k], lr[j, :, k], -np.inf).max())
        sg = float(np.where(vc[j, :, k], lg[j, :, k], -np.inf).max())
        info["ch"][ch] = {"true": y, "p_real": float(1 / (1 + np.exp(-sr))), "p_gen": float(1 / (1 + np.exp(-sg))),
                          "flag_real": sr >= thr[ch], "flag_gen": sg >= thr[ch]}
    out.append(info)
(P / "sheet_info.json").write_text(json.dumps(out, indent=1))
for i in out:
    print(i["row"], i["mode"], i["depth"], i["quadrant"], i["kind"], "cos a %.3f w %.3f" % (i["cos"]["agent"], i["cos"]["wrist"]),
          {c: ("T" if v["true"] else "F", round(v["p_real"], 2), round(v["p_gen"], 2)) for c, v in i["ch"].items()})
