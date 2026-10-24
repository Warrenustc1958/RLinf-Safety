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

"""Step 3: frame / action / label / latent alignment checks before batch caching.

Structural checks over all grader-labelled episodes (CPU), pixel seam checks on
sampled branches, gripper-sign check, VAE checks on the GPU (preprocessing,
causal context slots, dual split, batch invariance) and per-example window
sheets (Q1 root, Q2 / Q4 branches, c_grasp, tail, t0 = 0) around branch points,
first violations and first successes.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from toolkits.libero_safety.sam_alignment.cache_real import VAE, encode, load_dual_frames, load_vae
from toolkits.libero_safety.sam_alignment.schema import CHANNELS, CHUNK, OUT_ROOT, chunk_starts, chunk_window


def structural(eps):
    c = collections.Counter()
    for e in eps:
        seg_total = sum(b - a for _, a, b in e["agent_segments"])
        with np.load(e["labels"]) as z:
            T = len(z["cost"])
            off = z["official_violation"]
        c["labels_len_eq_num_frames_minus_1"] += T == e["num_frames"] - 1
        c["frames_ge_num_frames"] += seg_total >= e["num_frames"]
        c["wrist_segments_eq_agent"] += e["wrist_segments"] == e["agent_segments"]
        if e["kind"] != "root":
            bt = int(e["branch_timestep"])
            c["branches"] += 1
            c["branch_aligned_(bt-1)%8==0"] += (bt - 1) % CHUNK == 0
            c["branch_fingerprint_zero"] += e["state_fingerprint_maxdiff"] == 0.0
            c["branch_scene_mutated"] += bool(e["scene_mutated"])
            c["branch_root_in_index"] += bool(e["root_uid"])
            c["branch_prefix_len_eq_bt-1"] += e["agent_segments"][0][2] == bt - 1
        c["episodes"] += 1
        c["episodes_with_official_violation"] += bool(off.any())
    return dict(c)


def seam(eps, rng, n):
    br = [e for e in eps if e["kind"] != "root"]
    diffs = collections.defaultdict(list)
    for e in [br[i] for i in rng.choice(len(br), size=min(n, len(br)), replace=False)]:
        (rp, _, p), (bp, _, _) = e["agent_segments"][:2]
        with np.load(rp) as r, np.load(bp) as b:
            for key in ("agentview_image", "robot0_eye_in_hand_image"):
                diffs[key].append(float(np.abs(r[key][p].astype(np.int16) - b[key][0].astype(np.int16)).mean()))
                diffs[key + "_vs_next"].append(float(np.abs(r[key][p - 1].astype(np.int16)
                                                          - b[key][0].astype(np.int16)).mean()) if p > 0 else 0.0)
    return {k: {"mean": float(np.mean(v)), "max": float(np.max(v)), "n": len(v)} for k, v in diffs.items()}


def gripper_sign(eps, rng, n):
    """Sign agreement between the gripper command and the next finger-width change."""
    agree, total, keys = 0, 0, set()
    for e in [eps[i] for i in rng.choice(len(eps), size=min(n, len(eps)), replace=False)]:
        traj = json.load(open(e["json"]))["trajectory"]
        obs = [s.get("observation") or {} for s in traj]
        keys |= set(obs[0].keys())
        name = next((k for k in obs[0] if "gripper_qpos" in k), None)
        if name is None:
            continue
        width = np.array([np.abs(np.asarray(o[name], np.float32)).sum() for o in obs])
        cmd = np.array([s["action"][-1] for s in traj], np.float32)
        dw = np.diff(width)
        m = np.abs(dw) > 1e-4
        # +1 closes (width shrinks), -1 opens: agreement = sign(cmd) == -sign(dw)
        agree += int((np.sign(cmd[:-1][m]) == -np.sign(dw[m])).sum())
        total += int(m.sum())
    return {"agree": agree, "moving_steps": total, "rate": agree / max(total, 1), "obs_keys": sorted(keys)[:12]}


def vae_checks(pipe, eps, device, rng):
    out = collections.defaultdict(list)
    picks = [eps[i] for i in rng.choice(len(eps), size=4, replace=False)]
    wins = []
    for e in picks:
        frames = load_dual_frames(e)
        t0 = chunk_starts(e["num_frames"])[min(3, len(chunk_starts(e["num_frames"])) - 1)]
        wins.append(frames[chunk_window(t0, e["num_frames"])[0]])
    x = np.stack(wins)
    # preprocessing equivalence with the pipeline
    ref = pipe.preprocess_video([Image.fromarray(f) for f in x[0]])
    mine = torch.from_numpy(x[0]).permute(3, 0, 1, 2).float().div(127.5).sub(1)
    out["preprocess_maxabs"].append(float((ref[0].float().cpu() - mine).abs().max()))
    z13 = encode(pipe, x, device).float()
    z5 = encode(pipe, x[:, :5], device).float()
    out["latent_shape_13"] = list(z13.shape)
    out["latent_shape_5"] = list(z5.shape)
    out["ctx_13_vs_5_maxabs"].append(float((z13[:, :, :2] - z5).abs().max()))
    out["ctx_13_vs_5_rel"].append(float((z13[:, :, :2] - z5).norm() / z5.norm()))
    za = encode(pipe, x[:, :, :256], device).float()
    out["dual_top_vs_agent_only_maxabs"].append(float((z13[:, :, :, :16] - za).abs().max()))
    z1 = torch.cat([encode(pipe, x[i:i + 1], device).float() for i in range(len(x))])
    out["batch_vs_single_maxabs"].append(float((z13 - z1).abs().max()))
    out["latent_absmax"].append(float(z13.abs().max()))
    return dict(out)


def sheet(e, t0s, path):
    frames = load_dual_frames(e)
    with np.load(e["labels"]) as z:
        cost, valid, succ = z["cost"], z["valid_channel"], z["success"]
    tw, th = 64, 128
    rows = []
    for t0 in t0s:
        ids, vlen = chunk_window(t0, e["num_frames"])
        canvas = Image.new("RGB", (tw * len(ids), th + 30), "white")
        d = ImageDraw.Draw(canvas)
        for j, f in enumerate(ids):
            canvas.paste(Image.fromarray(frames[f]).resize((tw, th)), (j * tw, 0))
            d.text((j * tw + 2, th + 1), f"f{f}", fill="black")
            if j >= 5:  # future frame f = t + 1 carries labels of action t
                t = f - 1
                if j - 5 < vlen:
                    bits = "".join("1" if cost[t, k] else ("0" if valid[t, k] else "?") for k in range(len(CHANNELS)))
                    d.text((j * tw + 2, th + 14), bits + ("S" if succ[t] else ""),
                           fill="red" if cost[t].any() else "black")
        d.text((2, 2), f"t0={t0}", fill="yellow")
        rows.append(canvas)
    img = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        img.paste(r, (0, y))
        y += r.height
    img.save(path)


def examples(eps):
    def first(pred):
        return next((e for e in eps if pred(e)), None)
    ex = {
        "q1_root": first(lambda e: e["kind"] == "root" and e["actual_quadrant"] == "Q1"),
        "q2_branch": first(lambda e: e["kind"] != "root" and e["actual_quadrant"] == "Q2"),
        "q4_branch_obj": first(lambda e: e["kind"] != "root" and e["actual_quadrant"] == "Q4"
                               and e["first_event_by_channel"]["c_obj"] is not None),
        "q4_branch_robot": first(lambda e: e["kind"] != "root" and e["actual_quadrant"] == "Q4"
                                 and e["first_event_by_channel"]["c_robot"] is not None),
        "grasp": first(lambda e: e["first_event_by_channel"]["c_grasp"] is not None),
        "tail": first(lambda e: (e["num_frames"] - 1) % CHUNK != 0 and e["success"]),
    }
    return {k: v for k, v in ex.items() if v is not None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", default=str(OUT_ROOT / "labels"))
    ap.add_argument("--registry", default=str(OUT_ROOT / "splits" / "split_registry.json"))
    ap.add_argument("--out", default=str(OUT_ROOT / "reports" / "alignment"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    roles = json.loads(Path(args.registry).read_text())["episode_role"]
    eps = [json.loads(l) for l in (Path(args.labels) / "episodes.jsonl").read_text().splitlines()]
    eps = [e for e in eps if e["label_source"] == "grader_v1" and e["dual"]]
    train = [e for e in eps if roles.get(e["uid"]) == "cost_train"]

    rep = {"structural": structural(eps), "pixel_seam": seam(eps, rng, 120),
           "gripper_sign": gripper_sign(train, rng, 60)}
    device = "cuda:0"
    pipe = load_vae(VAE, device)
    rep["vae"] = vae_checks(pipe, train, device, rng)

    ex = examples(train)
    rep["examples"] = {}
    for name, e in ex.items():
        starts = chunk_starts(e["num_frames"])
        marks = {0}
        if e["kind"] != "root":
            marks.add(int(e["branch_timestep"]) - 1)
        for t in list(e["first_event_by_channel"].values()):
            if t is not None:
                marks.add(t // CHUNK * CHUNK)
        if e["success"]:
            with np.load(e["labels"]) as z:
                marks.add(int(np.flatnonzero(z["success"])[0]) // CHUNK * CHUNK)
        marks.add(starts[-1])
        t0s = sorted(m for m in marks if m in starts)
        sheet(e, t0s, out / f"{name}.png")
        rep["examples"][name] = {"uid": e["uid"], "kind": e["kind"], "quadrant": e["actual_quadrant"],
                                 "branch_timestep": e["branch_timestep"], "num_frames": e["num_frames"],
                                 "first_event_by_channel": e["first_event_by_channel"], "t0s": t0s,
                                 "sheet": str(out / f"{name}.png")}
    rep["notes"] = {
        "image_orientation": "frames used exactly as stored in the audit NPZ, the convention Wan was trained on",
        "sheet_legend": "under future frames: c_obj c_robot c_grasp (1 violation, 0 safe, ? invalid); S = success",
    }
    (out / "alignment_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("structural", "pixel_seam", "gripper_sign", "vae")}, indent=1))


if __name__ == "__main__":
    main()
