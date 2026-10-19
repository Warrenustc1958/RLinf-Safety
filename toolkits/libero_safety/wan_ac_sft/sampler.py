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

"""Window sampler and dataset for Wan AC-SFT over the S0 index.

A sample is ``num_frames`` frames: frame 0 of the episode (the clean anchor) plus
``num_frames - 1`` consecutive frames, matching ``RLinfNpyDataset``. Episodes are
drawn quadrant-balanced, then a window start is drawn:
  * ``template_frac``: the episode start template ``[0] * 5 + [1..8]``;
  * ``event_frac``: a window covering a violation (+-``event_radius``) or the branch point;
  * otherwise uniform over every valid start (the whole episode, not only the first 262 frames).
For branch episodes, uniform draws land in the inherited root prefix only with
probability ``prefix_keep`` (the prefix duplicates the root's own windows).

Actions follow the WM backend convention ``action[f]`` = the action that produced
frame ``f``: ``action[f] = audit_action[f - 1]`` and frame 0 gets ``[0, ..., 0, -1]``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

QUADRANTS = ("Q1", "Q2", "Q3", "Q4")
START_TEMPLATE = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
NULL_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], np.float32)


def load_manifest(index_dir: str | Path, split: str, dual_only: bool = False) -> list[dict]:
    rows = [json.loads(l) for l in (Path(index_dir) / "manifest.jsonl").read_text().splitlines()]
    return [r for r in rows if r["split"] == split and (r["dual"] or not dual_only)]


class WindowSampler:
    """Draws (row, frame_ids, mode) without touching image data."""

    def __init__(
        self,
        rows: list[dict],
        num_frames: int = 13,
        quadrant_weights: dict[str, float] | None = None,
        event_frac: float = 0.5,
        event_radius: int = 16,
        template_frac: float = 0.05,
        prefix_keep: float = 0.25,
    ):
        self.num_frames = num_frames
        self.span = num_frames - 1
        self.rows = [r for r in rows if r["num_frames"] >= num_frames]
        self.dropped_short = len(rows) - len(self.rows)
        self.by_q = {q: [r for r in self.rows if r["quadrant"] == q] for q in QUADRANTS}
        present = [q for q in QUADRANTS if self.by_q[q]]
        w = quadrant_weights or {q: 1.0 for q in present}
        total = sum(w.get(q, 0.0) for q in present)
        self.quadrants = present
        self.q_probs = np.array([w.get(q, 0.0) / total for q in present])
        self.event_frac = event_frac
        self.event_radius = event_radius
        self.template_frac = template_frac
        self.prefix_keep = prefix_keep

    @staticmethod
    def events(row: dict) -> list[int]:
        return [f for f in (row["first_violation_frame"], row["branch_frame"]) if f]

    def draw(self, rng: np.random.Generator) -> tuple[dict, list[int], str]:
        q = self.quadrants[rng.choice(len(self.quadrants), p=self.q_probs)]
        row = self.by_q[q][rng.integers(len(self.by_q[q]))]
        n = row["num_frames"]
        hi = n - self.span  # starts in [1, hi]; window = [0] + [s, s + span)
        u = rng.random()
        if u < self.template_frac:
            return row, list(START_TEMPLATE), "template"
        ev = self.events(row)
        if ev and u < self.template_frac + self.event_frac:
            c = int(ev[rng.integers(len(ev))]) + int(rng.integers(-self.event_radius, self.event_radius + 1))
            s = int(np.clip(c - self.span // 2, 1, hi))
            mode = "event"
        else:
            bf = row["branch_frame"]
            if bf and bf > 1 and rng.random() > self.prefix_keep:
                lo = min(max(1, bf - self.span // 2), hi)
            else:
                lo = 1
            s = int(rng.integers(lo, hi + 1))
            mode = "uniform"
        return row, [0] + list(range(s, s + self.span)), mode


def window_actions(actions: np.ndarray, frame_ids: list[int]) -> np.ndarray:
    """Per-frame actions in the backend convention (action that produced the frame)."""
    out = np.empty((len(frame_ids), actions.shape[1]), np.float32)
    for i, f in enumerate(frame_ids):
        out[i] = NULL_ACTION if f == 0 or i == 0 else actions[f - 1]
    return out


def load_frames(segments: list[list], frame_ids: list[int], key: str) -> np.ndarray:
    """Gather stitched frames from ``[npz, start, stop]`` segments."""
    bounds = np.cumsum([0] + [b - a for _, a, b in segments])
    need: dict[int, list[tuple[int, int]]] = {}
    for i, f in enumerate(frame_ids):
        seg = int(np.searchsorted(bounds, f, side="right")) - 1
        need.setdefault(seg, []).append((i, segments[seg][1] + f - int(bounds[seg])))
    out = [None] * len(frame_ids)
    for seg, items in need.items():
        with np.load(segments[seg][0]) as z:
            arr = z[key]
            for i, local in items:
                out[i] = arr[local]
    return np.stack(out)


class WanACSFTDataset:
    """Map-style dataset; item ``i`` is a reproducible random window."""

    def __init__(self, index_dir, split="train", view="agent", length=100_000, seed=0, **sampler_kw):
        if view not in ("agent", "dual"):
            raise ValueError(f"view must be agent|dual, got {view}")
        self.view = view
        self.rows = load_manifest(index_dir, split, dual_only=(view == "dual"))
        self.sampler = WindowSampler(self.rows, **sampler_kw)
        self.length = length
        self.seed = seed

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        from PIL import Image  # local import keeps the index tools dependency-free

        rng = np.random.default_rng([self.seed, i])
        row, frame_ids, _ = self.sampler.draw(rng)
        with np.load(row["labels"]) as lab:
            actions = window_actions(lab["actions"], frame_ids)
        agent = load_frames(row["agent_segments"], frame_ids, "agentview_image")
        item = {
            "video": [Image.fromarray(f) for f in agent],
            "reference_image": [Image.fromarray(agent[0])],
            "action": actions,
            "meta": {"idx": row["idx"], "quadrant": row["quadrant"], "frame_ids": frame_ids},
        }
        if self.view == "dual":
            wrist = load_frames(row["wrist_segments"], frame_ids, "robot0_eye_in_hand_image")
            item["wrist_video"] = [Image.fromarray(f) for f in wrist]
        return item
