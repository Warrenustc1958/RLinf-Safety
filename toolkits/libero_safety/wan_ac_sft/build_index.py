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

"""S0 for Wan AC-SFT: index audit episodes, stitch branch chains, split by init state.

Writes, under ``--out``:
  manifest.jsonl     one row per episode: stitched frame segments, metadata, split
  labels/<idx>.npz   stitched per-step arrays (actions, violation, severe, success,
                     eef, gripper, inherited)
  split.json         held-out (suite, level, task) -> {"val": [...], "test": [...]}

Frames are not copied. A row's ``agent_segments`` / ``wrist_segments`` are
``[npz_path, start, stop]`` slices whose concatenation is the stitched episode.

Stitching is recursive: ``branch_timestep`` is absolute from the root reset, and
a branch's parent can itself be a branch, so a branch is
``stitched(parent)[:branch_timestep - 1] + own``. The branch's first frame is the
restored state that replaces parent frame ``branch_timestep - 1``, so
``frame[t] + action[t] -> frame[t + 1]`` holds across every seam.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "libero_safety_wan_ac_sft_index/v1"
AGENT_KEY = "agentview_image"
WRIST_KEY = "robot0_eye_in_hand_image"
TRANSITION_KEYS = ("actions", "violation", "severe", "success")
FRAME_KEYS = ("eef", "gripper")


def parse_episode(path: str) -> dict[str, Any]:
    """Read one audit JSON into flat per-step arrays (own frames only)."""
    rec = json.loads(Path(path).read_text())
    ep = rec["episode"]
    traj = rec["trajectory"]
    vis = rec["visual_trajectory"]
    n_frames = int(vis["num_frames"])
    if n_frames != len(traj) + 1:
        raise ValueError(f"{path}: {n_frames} frames for {len(traj)} transitions")
    obs0 = rec.get("initial_observation") or {}
    first = obs0 if "robot0_eef_pos" in obs0 else traj[0]["observation"]
    frame_obs = [first] + [s["observation"] for s in traj]
    term = [s["terminations"] for s in traj]
    return {
        "json": path,
        "dir": str(Path(path).parent),
        "eid": ep["episode_id"],
        "parent": ep.get("root_episode_id") or rec.get("root_episode_id"),
        "bt": ep.get("branch_timestep"),
        "suite": ep["task_suite"],
        "level": ep["safety_level"],
        "task": int(ep["task_id"]),
        "trial": int(ep["trial_id"]),
        "reset_state_id": ep.get("reset_state_id"),
        "kind": ep.get("episode_kind", "root"),
        "quadrant": rec["outcome"]["quadrant"],
        "npz": str(Path(path).parent / vis["path"]),
        "n_frames": n_frames,
        "eef_frame0_from_initial": "robot0_eef_pos" in obs0,
        "actions": np.asarray([s["action"] for s in traj], np.float32),
        "violation": np.asarray([bool(t["safety_violation"]) for t in term]),
        "severe": np.asarray([bool(t["severe_collision"]) for t in term]),
        "success": np.asarray([bool(t["raw_task"]) for t in term]),
        "eef": np.asarray([o["robot0_eef_pos"] for o in frame_obs], np.float32),
        "gripper": np.asarray([o["robot0_gripper_qpos"] for o in frame_obs], np.float32),
    }


def take_frames(segments: list[list], n: int) -> list[list]:
    """First ``n`` frames of a segment list."""
    out, left = [], n
    for src, start, stop in segments:
        if left <= 0:
            break
        k = min(stop - start, left)
        out.append([src, start, start + k])
        left -= k
    if left:
        raise ValueError(f"segments hold fewer than {n} frames")
    return out


def stitch_all(parsed: dict[tuple[str, str], dict]) -> dict[tuple[str, str], dict]:
    """Stitch every episode onto its (possibly nested) ancestor chain."""
    memo: dict[tuple[str, str], dict] = {}

    def go(key, depth=0):
        if key in memo:
            return memo[key]
        if depth > 64:
            raise ValueError(f"branch chain deeper than 64 at {key}")
        e = parsed[key]
        own_segments = [[e["json"], 0, e["n_frames"]]]
        own = {k: e[k] for k in TRANSITION_KEYS + FRAME_KEYS}
        if e["parent"] is None or e["parent"] == e["eid"]:
            out = {
                "segments": own_segments,
                "arrays": own,
                "inherited": np.zeros(len(e["actions"]), bool),
                "chain": [e["json"]],
                "prefix": 0,
                "prefix_available": True,
            }
        else:
            pkey = (e["dir"], e["parent"])
            if pkey not in parsed:
                raise KeyError(f"{e['json']}: parent {e['parent']} is not in the index")
            if e["bt"] is None:
                raise ValueError(f"{e['json']}: branch without branch_timestep")
            p = go(pkey, depth + 1)
            pe = parsed[pkey]
            if (pe["suite"], pe["level"], pe["task"], pe["trial"]) != (
                e["suite"], e["level"], e["task"], e["trial"]
            ):
                raise ValueError(f"{e['json']}: init state differs from parent {pe['json']}")
            prefix = int(e["bt"]) - 1
            if prefix < 0:
                raise ValueError(f"{e['json']}: branch_timestep={e['bt']} < 1")
            if prefix > len(p["arrays"]["actions"]):
                # The branch state lies past the parent's saved transitions (the
                # snapshot was taken after the audit stopped), so its history
                # cannot be rebuilt: keep it as a stand-alone, keyframe-started episode.
                out = {
                    "segments": own_segments,
                    "arrays": own,
                    "inherited": np.zeros(len(e["actions"]), bool),
                    "chain": [e["json"]],
                    "prefix": 0,
                    "prefix_available": False,
                }
                memo[key] = out
                return out
            out = {
                "segments": take_frames(p["segments"], prefix) + own_segments,
                "arrays": {
                    k: np.concatenate([p["arrays"][k][:prefix], own[k]], axis=0)
                    for k in TRANSITION_KEYS + FRAME_KEYS
                },
                "inherited": np.concatenate(
                    [np.ones(prefix, bool), np.zeros(len(e["actions"]), bool)]
                ),
                "chain": p["chain"] + [e["json"]],
                "prefix": prefix,
                "prefix_available": p["prefix_available"],
            }
        memo[key] = out
        return out

    for key in parsed:
        go(key)
    return memo


def make_split(
    trials_by_task: dict[tuple, set[int]], n_val: int, n_test: int, seed: int
) -> dict[tuple, dict[str, list[int]]]:
    """Deterministically hold out ``n_val + n_test`` init states per task."""
    split = {}
    for task_key in sorted(trials_by_task):
        trials = sorted(trials_by_task[task_key])
        if len(trials) < n_val + n_test + 1:
            raise ValueError(f"{task_key}: only {len(trials)} init states")
        tag = f"{seed}|" + "|".join(map(str, task_key))
        rng = random.Random(int(hashlib.sha256(tag.encode()).hexdigest()[:12], 16))
        held = rng.sample(trials, n_val + n_test)
        split[task_key] = {"val": sorted(held[:n_val]), "test": sorted(held[n_val:])}
    return split


def _first(mask: np.ndarray) -> int | None:
    """Frame index showing the first True transition (transition t -> frame t+1)."""
    hit = np.flatnonzero(mask)
    return int(hit[0]) + 1 if hit.size else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kept", required=True, help="kept.txt with one audit JSON per line")
    ap.add_argument("--wrist-results", required=True, help="wrist re-render results.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-val", type=int, default=2)
    ap.add_argument("--n-test", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    paths = [str(Path(p).resolve()) for p in Path(args.kept).read_text().split()]
    wrist_ok: dict[str, str] = {}
    for line in Path(args.wrist_results).read_text().splitlines():
        r = json.loads(line)
        key = str(Path(r["json"]).resolve())
        if r["status"] == "ok":
            wrist_ok[key] = r["wrist"]
        else:
            wrist_ok.pop(key, None)

    with Pool(args.workers) as pool:
        parsed_list = pool.map(parse_episode, paths, chunksize=16)
    parsed = {(e["dir"], e["eid"]): e for e in parsed_list}
    if len(parsed) != len(parsed_list):
        raise ValueError("duplicate (dir, episode_id) keys in kept list")
    stitched = stitch_all(parsed)
    npz_of = {e["json"]: e["npz"] for e in parsed_list}

    trials_by_task: dict[tuple, set[int]] = defaultdict(set)
    for e in parsed_list:
        trials_by_task[(e["suite"], e["level"], e["task"])].add(e["trial"])
    split = make_split(trials_by_task, args.n_val, args.n_test, args.seed)

    out = Path(args.out)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    rows, counts = [], Counter()
    for idx, e in enumerate(sorted(parsed_list, key=lambda x: x["json"])):
        s = stitched[(e["dir"], e["eid"])]
        task_key = (e["suite"], e["level"], e["task"])
        part = next((k for k, v in split[task_key].items() if e["trial"] in v), "train")
        dual = all(j in wrist_ok for j in s["chain"])
        a = s["arrays"]
        n_frames = len(a["eef"])
        if n_frames != len(a["actions"]) + 1 or sum(b - a_ for _, a_, b in s["segments"]) != n_frames:
            raise ValueError(f"{e['json']}: stitched frames/actions disagree")
        label_path = out / "labels" / f"{idx:05d}.npz"
        np.savez(label_path, inherited=s["inherited"], **a)
        rows.append({
            "idx": idx,
            "json": e["json"],
            "episode_id": e["eid"],
            "suite": e["suite"],
            "level": e["level"],
            "cell": f"{e['suite']}_{e['level']}",
            "task": e["task"],
            "trial": e["trial"],
            "reset_state_id": e["reset_state_id"],
            "quadrant": e["quadrant"],
            "kind": e["kind"],
            "parent": e["parent"],
            "branch_timestep": e["bt"],
            "chain_depth": len(s["chain"]) - 1,
            "prefix_available": s["prefix_available"],
            "num_frames": n_frames,
            "branch_frame": int(s["inherited"].sum()) if len(s["chain"]) > 1 else None,
            "first_violation_frame": _first(a["violation"]),
            "first_success_frame": _first(a["success"]),
            "agent_segments": [[npz_of[j], st, sp] for j, st, sp in s["segments"]],
            "wrist_segments": [[wrist_ok[j], st, sp] for j, st, sp in s["segments"]] if dual else None,
            "dual": dual,
            "split": part,
            "labels": str(label_path),
        })
        counts[(part, e["quadrant"])] += 1
    with (out / "manifest.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out / "split.json").write_text(json.dumps(
        {"schema_version": SCHEMA_VERSION, "seed": args.seed,
         "held_out": {"|".join(map(str, k)): v for k, v in split.items()}}, indent=1))
    print(f"episodes={len(rows)} dual={sum(r['dual'] for r in rows)} "
          f"nested_branches={sum(r['chain_depth'] > 1 for r in rows)} "
          f"standalone_branches={sum(r['kind'] != 'root' and not r['prefix_available'] for r in rows)} "
          f"eef_frame0_fallback={sum(not e['eef_frame0_from_initial'] for e in parsed_list)}")
    for part in ("train", "val", "test"):
        print(part, {q: counts[(part, q)] for q in ("Q1", "Q2", "Q3", "Q4")})


if __name__ == "__main__":
    main()
