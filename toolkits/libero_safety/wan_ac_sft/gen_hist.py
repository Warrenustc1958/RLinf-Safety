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

"""T3 (simulator forcing): train on Wan's own open-loop histories with the recorded future as the target.

``CodeWorld-Safe/.../scgrpo/drift/t0_gen.py`` rolls dual-41k out open loop on every recorded episode and stores the
4 generated history frames before every chunk d >= 1. A generated item is the WM-backend window of chunk d,
frames [0, 8d-3 .. 8d+8], with frames 1-4 (the history, latent slot 1) replaced by the generated ones; the 8 future
frames (latent slots 2-3, the only ones in the loss) stay the recorded ones. With probability 1 - p an ordinary
real-history window is drawn instead (the teacher-forced recipe), so teacher-forced quality is kept.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from toolkits.libero_safety.wan_ac_sft.sampler import load_frames, window_actions

CHUNK = 8


def _dec(buf, off, i):
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(buf[off[i]:off[i + 1]].tobytes())).convert("RGB"))


class GenHistItems:
    """Mixes generated-history windows (probability ``p``) into an inner WanACSFTDataset (dual view)."""

    def __init__(self, inner, t0_dir, split, p=0.7, seed=0, deterministic=False):
        self.inner, self.p, self.seed, self.det = inner, float(p), seed, deterministic
        self.view = inner.view
        self.dir = Path(t0_dir) / split
        have = {f.stem for f in self.dir.glob("*.npz")}  # episodes whose generation has finished
        self.rows = [r for r in inner.rows if r["idx"] in have]
        self.motion = inner.sampler.motion
        if not self.rows:
            raise ValueError(f"no generated histories in {self.dir}")

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i, 31])
        if not self.det and rng.random() >= self.p:
            return self.inner[i]
        row = self.rows[rng.integers(len(self.rows))]
        z = np.load(self.dir / f"{row['idx']}.npz")
        D = int(z["n_chunks"])
        if D < 2:
            return self.inner[i]
        ds = np.arange(1, D)
        w = None
        prof = self.motion(row) if self.motion is not None else None
        if prof is not None:  # same motion weighting as the real sampler, at the chunk's first predicted frame
            starts = CHUNK * ds - 3
            ok = starts < len(prof)
            if ok.all():
                w = np.maximum(np.nan_to_num(prof[starts], nan=self.motion.floor), self.motion.floor) ** self.motion.power
                w = w / w.sum()
        d = int(rng.choice(ds, p=w))
        ids = [0] + list(range(CHUNK * d - 3, CHUNK * d + CHUNK + 1))
        agent = load_frames(row["agent_segments"], ids, "agentview_image")
        wrist = load_frames(row["wrist_segments"], ids, "robot0_eye_in_hand_image")
        hist = np.stack([_dec(z["hist_jpg"], z["hist_off"], 4 * (d - 1) + k) for k in range(4)])
        agent[1:5], wrist[1:5] = hist[:, :256], hist[:, 256:]
        with np.load(row["labels"]) as lab:
            act = window_actions(lab["actions"], ids, self.inner.pose_scale, self.inner.pose_clip)
        from PIL import Image

        return {"video": [Image.fromarray(f) for f in agent], "reference_image": [Image.fromarray(agent[0])],
                "wrist_video": [Image.fromarray(f) for f in wrist], "action": act,
                "meta": {"idx": row["idx"], "quadrant": row["quadrant"], "frame_ids": ids, "gen_chunk": d}}
