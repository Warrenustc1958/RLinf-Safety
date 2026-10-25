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

"""P0: same-state counterfactual pairs for the Step 6 ranking loss, with hardness from B0.

A pair is two real-cache chunks of one role at the same branch point (same root_uid,
t0 = bt - 1), both from eligible episodes (root, or branch with a recorded zero state
fingerprint and no scene mutation), with the same valid length, where channel k is a
known positive in one and a known negative in the other.

Hardness: B0 (task + action only) margin S_unsafe - S_safe on the channel's max valid
logit. ``hard`` = B0 orders the pair wrongly or with margin < ``--hard_margin``. The
report gives B0 / B1 / B2 pair accuracy on all vs hard pairs for the eval roles.

Writes <out>/<role>_pairs.npz and <out>/report.json.
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
from toolkits.libero_safety.sam_alignment.train import chunk_labels


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = SAMNet(**ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    return model.cuda().eval()


@torch.no_grad()
def logits_of(model, meta, rows, z_path=None, bs=256):
    out = []
    zmm = np.load(z_path, mmap_mode="r") if z_path is not None else None
    for s in range(0, len(rows), bs):
        r = rows[s:s + bs]
        a = torch.from_numpy(meta["action"][r]).cuda()
        t = torch.from_numpy(meta["task_id"][r].astype(np.int64)).cuda()
        z = (torch.from_numpy(np.asarray(zmm[r], np.float32)).cuda() if zmm is not None
             else torch.zeros(len(r), 1, device="cuda"))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, _ = model(z, a, t)
        out.append(lg.float().cpu().numpy())
    return np.concatenate(out)


def eligible_episodes(eps):
    ok = {}
    for e in eps:
        if e["kind"] == "root":
            ok[e["uid"]] = True
        else:
            ok[e["uid"]] = (e.get("state_fingerprint_maxdiff") is not None
                            and float(e["state_fingerprint_maxdiff"]) == 0.0 and not e.get("scene_mutated"))
    return ok


def build_pairs(meta, ep_of, ok, reason):
    rows = np.flatnonzero(meta["is_pair"])
    pos, known, _, _ = chunk_labels(meta, rows, np.zeros(meta["cost"][rows].shape, np.float32))
    vlen = meta["valid_step"][rows].sum(1)
    groups = collections.defaultdict(list)
    for n, r in enumerate(rows):
        e = ep_of.get(str(meta["uid"][r]))
        if e is None:
            reason["episode_not_in_labels"] += 1
            continue
        if not ok[e["uid"]]:
            reason["branch_no_fingerprint_or_mutated"] += 1
            continue
        groups[(e["root_uid"] or e["uid"], int(meta["t0"][r]))].append(n)
    pairs = []
    for (_, _), members in groups.items():
        if len(members) < 2:
            reason["group_singleton"] += 1
            continue
        for i in members:
            for j in members:
                if i == j or vlen[i] != vlen[j]:
                    continue
                for k in range(len(CHANNELS)):
                    if known[i, k] and known[j, k] and pos[i, k] and not pos[j, k]:
                        pairs.append((rows[i], rows[j], k))
    return np.array(pairs, np.int64).reshape(-1, 3), len(groups)


def chunk_score(logits, meta, rows, k):
    vc = meta["valid_channel"][rows][..., k]
    return np.where(vc, logits[..., k], -np.inf).max(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default=str(OUT_ROOT / "cache" / "real_v1"))
    ap.add_argument("--labels", default=str(OUT_ROOT / "labels" / "episodes.jsonl"))
    ap.add_argument("--ckpt", default=str(OUT_ROOT / "checkpoints"))
    ap.add_argument("--out", default=str(OUT_ROOT / "pairs" / "rank_pairs_v1"))
    ap.add_argument("--roles", nargs="+", default=["cost_train", "h_train", "dev", "diag_newtest"])
    ap.add_argument("--eval_roles", nargs="+", default=["dev", "diag_newtest"])
    ap.add_argument("--hard_margin", type=float, default=1.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eps = [json.loads(l) for l in Path(args.labels).read_text().splitlines()]
    ep_of = {e["uid"]: e for e in eps}
    ok = eligible_episodes(eps)
    ck = Path(args.ckpt)
    models = {"B0": load_model(ck / "cost_real_B0" / "best.pt")}
    for v in ("B1", "B2"):
        models[v] = load_model(ck / f"cost_real_{v}" / "best.pt")
    report = {"hard_margin": args.hard_margin, "roles": {}}
    cache = Path(args.cache)
    for role in args.roles:
        with np.load(cache / f"{role}_meta.npz") as m:
            meta = {k: m[k] for k in m.files}
        reason = collections.Counter()
        pairs, n_groups = build_pairs(meta, ep_of, ok, reason)
        rows = np.unique(pairs[:, :2]) if len(pairs) else np.zeros(0, np.int64)
        pos_of = {int(r): n for n, r in enumerate(rows)}
        iu = np.array([pos_of[int(r)] for r in pairs[:, 0]], np.int64)
        isf = np.array([pos_of[int(r)] for r in pairs[:, 1]], np.int64)
        names = ["B0"] + (["B1", "B2"] if role in args.eval_roles else [])
        margins = {}
        for v in names:
            lg = logits_of(models[v], meta, rows, None if v == "B0" else cache / f"{role}_z.npy")
            mg = np.zeros(len(pairs), np.float32)
            for k in range(len(CHANNELS)):
                sel = pairs[:, 2] == k
                if sel.any():
                    s = chunk_score(lg, meta, rows, k)
                    mg[sel] = s[iu[sel]] - s[isf[sel]]
            margins[v] = mg
        a = meta["action"]
        act_sd = models["B0"].act_sd.cpu().numpy()
        act_dist = np.linalg.norm((a[pairs[:, 0]] - a[pairs[:, 1]]) / act_sd, axis=-1).mean(1)
        hard = margins["B0"] < args.hard_margin
        np.savez(out / f"{role}_pairs.npz", unsafe_row=pairs[:, 0], safe_row=pairs[:, 1], channel=pairs[:, 2],
                 b0_margin=margins["B0"], act_dist=act_dist.astype(np.float32), hard=hard,
                 root_uid=np.array([ep_of[str(meta["uid"][r])]["root_uid"] or str(meta["uid"][r])
                                    for r in pairs[:, 0]]))
        rr = {"pair_groups": n_groups, "skipped": dict(reason), "pairs": int(len(pairs)), "by_channel": {}}
        for k, ch in enumerate(CHANNELS):
            sel = pairs[:, 2] == k
            c = {"pairs": int(sel.sum()), "hard": int((sel & hard).sum()),
                 "roots": int(len({str(meta["uid"][r]) for r in pairs[sel, 0]})) if sel.any() else 0,
                 "act_dist_median": float(np.median(act_dist[sel])) if sel.any() else None}
            for v, mg in margins.items():
                c[f"{v}_acc_all"] = float((mg[sel] > 0).mean()) if sel.any() else None
                c[f"{v}_acc_hard"] = float((mg[sel & hard] > 0).mean()) if (sel & hard).any() else None
            rr["by_channel"][ch] = c
        report["roles"][role] = rr
        print(role, json.dumps(rr), flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
