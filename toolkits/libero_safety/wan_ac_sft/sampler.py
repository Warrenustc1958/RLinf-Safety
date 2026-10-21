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
  * otherwise over every valid start (the whole episode, not only the first 262 frames),
    weighted by ``motion_weight`` (see below).
For branch episodes, uniform draws land in the inherited root prefix only with
probability ``prefix_keep`` (the prefix duplicates the root's own windows).

Motion weighting (``build_motion_index.py`` writes the profiles): a plain uniform
draw puts 33-56 % of windows on spans whose 8 predicted frames move less than
2-5/255, i.e. on almost-static video where "copy the memory frames" is the
loss-optimal answer and the action carries no signal. With a profile available
the start is drawn with probability proportional to
``max(motion[s], motion_floor) ** motion_power`` over the allowed starts, where
``motion[s]`` is the mean absolute pixel change of the predicted span in 0-255
units. Episodes without a profile fall back to uniform.

Actions follow the WM backend convention ``action[f]`` = the action that produced
frame ``f``: ``action[f] = audit_action[f - 1]`` and frame 0 gets ``[0, ..., 0, -1]``.
``pose_scale`` multiplies the six pose dims before they reach the model: their
mean magnitude (about 0.02) is an order of magnitude below the gripper (about
1.0), so unscaled pose commands are drowned out by the gripper in the action
embedding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

QUADRANTS = ("Q1", "Q2", "Q3", "Q4")
START_TEMPLATE = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
NULL_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], np.float32)
POSE_DIMS = 6  # the gripper stays on its own scale


def load_manifest(index_dir: str | Path, split: str, dual_only: bool = False) -> list[dict]:
    rows = [json.loads(l) for l in (Path(index_dir) / "manifest.jsonl").read_text().splitlines()]
    return [r for r in rows if r["split"] == split and (r["dual"] or not dual_only)]


class MotionProfiles:
    """Lazily loaded per-episode motion profiles (``<index>/motion/<idx>.npy``)."""

    def __init__(self, index_dir: str | Path | None, floor: float = 0.5, power: float = 1.0):
        self.dir = Path(index_dir) / "motion" if index_dir else None
        self.floor = float(floor)
        self.power = float(power)
        self._cache: dict[str, np.ndarray | None] = {}
        self.hits = 0

    def __call__(self, row: dict) -> np.ndarray | None:
        idx = row["idx"]
        if idx in self._cache:
            return self._cache[idx]
        prof = None
        if self.dir is not None:
            path = self.dir / f"{idx}.npy"
            if path.exists():
                try:
                    prof = np.load(path)
                except Exception:
                    prof = None
        if prof is not None:
            self.hits += 1
        self._cache[idx] = prof
        return prof

    def weights(self, row: dict, lo: int, hi: int) -> np.ndarray | None:
        """Start weights for ``s in [lo, hi]`` (inclusive), or ``None`` for uniform."""
        prof = self(row)
        if prof is None:
            return None
        lo, hi = max(lo, 1), min(hi, len(prof) - 1)
        if hi < lo:
            return None
        w = prof[lo: hi + 1].astype(np.float64)
        bad = ~np.isfinite(w)
        if bad.any():
            fill = np.median(w[~bad]) if (~bad).any() else self.floor
            w[bad] = fill
        w = np.maximum(w, self.floor) ** self.power
        total = w.sum()
        return w / total if total > 0 else None


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
        motion: MotionProfiles | None = None,
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
        self.motion = motion

    @staticmethod
    def events(row: dict) -> list[int]:
        return [f for f in (row["first_violation_frame"], row["branch_frame"]) if f]

    def _draw_start(self, row: dict, rng: np.random.Generator, lo: int, hi: int) -> int:
        """Start in ``[lo, hi]`` (inclusive), motion-weighted when a profile exists."""
        if self.motion is not None:
            w = self.motion.weights(row, lo, hi)
            if w is not None:
                return int(lo + rng.choice(w.size, p=w))
        return int(rng.integers(lo, hi + 1))

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
            s = self._draw_start(row, rng, lo, hi)
            mode = "uniform"
        return row, [0] + list(range(s, s + self.span)), mode


def scale_pose(actions: np.ndarray, pose_scale: float = 1.0, pose_clip: float = 0.0) -> np.ndarray:
    """Pose scaling (and optional cap) for actions on their way to the model.

    ``pose_clip`` caps the scaled pose in the model's action units; a handful of
    steps (0.15 % at a scale of 8) carry fast motions whose scaled magnitude
    would otherwise exceed the gripper's range of +-1 by several times.
    """
    if pose_scale == 1.0:
        return actions
    out = np.array(actions, np.float32, copy=True)
    out[..., :POSE_DIMS] *= pose_scale
    if pose_clip > 0:
        np.clip(out[..., :POSE_DIMS], -pose_clip, pose_clip, out=out[..., :POSE_DIMS])
    return out


def window_actions(actions: np.ndarray, frame_ids: list[int], pose_scale: float = 1.0,
                   pose_clip: float = 0.0) -> np.ndarray:
    """Per-frame actions in the backend convention (action that produced the frame)."""
    out = np.empty((len(frame_ids), actions.shape[1]), np.float32)
    for i, f in enumerate(frame_ids):
        out[i] = NULL_ACTION if f == 0 or i == 0 else actions[f - 1]
    return scale_pose(out, pose_scale, pose_clip)


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

    def __init__(self, index_dir, split="train", view="agent", length=100_000, seed=0,
                 motion_weight=True, motion_power=1.0, motion_floor=0.5, pose_scale=1.0,
                 pose_clip=0.0, **sampler_kw):
        if view not in ("agent", "dual"):
            raise ValueError(f"view must be agent|dual, got {view}")
        self.view = view
        self.rows = load_manifest(index_dir, split, dual_only=(view == "dual"))
        motion = MotionProfiles(index_dir if motion_weight else None,
                                floor=motion_floor, power=motion_power) if motion_weight else None
        self.sampler = WindowSampler(self.rows, motion=motion, **sampler_kw)
        self.length = length
        self.seed = seed
        self.pose_scale = float(pose_scale)
        self.pose_clip = float(pose_clip)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        from PIL import Image  # local import keeps the index tools dependency-free

        rng = np.random.default_rng([self.seed, i])
        row, frame_ids, _ = self.sampler.draw(rng)
        with np.load(row["labels"]) as lab:
            actions = window_actions(lab["actions"], frame_ids, self.pose_scale, self.pose_clip)
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
