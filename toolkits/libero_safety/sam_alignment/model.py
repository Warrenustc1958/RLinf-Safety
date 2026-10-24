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

"""SAM base cost model (Step 4 minimal architecture).

Input latent z [B, 48, 4, 32, 16]: slots 0 anchor, 1 memory (z_ctx), 2-3 future
(z_future); rows 0:16 agent, 16:32 wrist. Each view / slot goes through a shared
conv stem (48->128 stride 2, 128->256 stride 1) into 8x8 tokens, plus view, slot
role, 2D position embeddings. The 8 executed actions (standardized) are tokens
with a step embedding, plus one task token. A 2-layer transformer mixes all
tokens; the action tokens give per-step cost logits [8, 3] and proximity [8, 2].

Variants (same trunk, same budget):
  B0  task + action                 (no visual tokens)
  B1  real context + action         (slots 0, 1)
  B2  real context + future + action (slots 0-3)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from toolkits.libero_safety.sam_alignment.schema import CHUNK, K, PROX_CHANNELS

VARIANT_SLOTS = {"B0": (), "B1": (0, 1), "B2": (0, 1, 2, 3)}


class SAMNet(nn.Module):
    def __init__(self, variant="B2", views=2, num_tasks=15, dim=256, heads=8, layers=2,
                 latent_mean=None, latent_std=None, action_mean=None, action_std=None, dropout=0.1):
        super().__init__()
        if variant not in VARIANT_SLOTS:
            raise ValueError(f"unknown variant {variant}")
        self.variant, self.views, self.num_tasks, self.dim = variant, views, num_tasks, dim
        self.slots = list(VARIANT_SLOTS[variant])
        self.register_buffer("lat_mu", torch.tensor(latent_mean or [0.0] * 48).view(1, 48, 1, 1, 1))
        self.register_buffer("lat_sd", torch.tensor(latent_std or [1.0] * 48).view(1, 48, 1, 1, 1))
        self.register_buffer("act_mu", torch.tensor(action_mean or [0.0] * 7))
        self.register_buffer("act_sd", torch.tensor(action_std or [1.0] * 7))
        if self.slots:
            self.stem = nn.Sequential(nn.Conv2d(48, 128, 3, 2, 1), nn.GELU(), nn.Conv2d(128, dim, 3, 1, 1))
            self.view_emb = nn.Parameter(torch.zeros(2, dim))
            self.slot_emb = nn.Parameter(torch.zeros(4, dim))
            self.pos_emb = nn.Parameter(torch.zeros(64, dim))
        self.act = nn.Sequential(nn.Linear(7, dim), nn.GELU(), nn.Linear(dim, dim))
        self.step_emb = nn.Parameter(torch.zeros(CHUNK, dim))
        self.task_emb = nn.Embedding(num_tasks, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(dim)
        self.cost_head = nn.Linear(dim, K)
        self.prox_head = nn.Linear(dim, len(PROX_CHANNELS))
        for p in (self.step_emb, *( [self.view_emb, self.slot_emb, self.pos_emb] if self.slots else [])):
            nn.init.trunc_normal_(p, std=0.02)

    def config(self) -> dict:
        return {"variant": self.variant, "views": self.views, "num_tasks": self.num_tasks, "dim": self.dim,
                "latent_mean": self.lat_mu.flatten().tolist(), "latent_std": self.lat_sd.flatten().tolist(),
                "action_mean": self.act_mu.tolist(), "action_std": self.act_sd.tolist()}

    def visual_tokens(self, z):
        z = (z - self.lat_mu) / self.lat_sd
        b, s = z.shape[0], len(self.slots)
        out = []
        for v in range(self.views):
            zv = z[:, :, self.slots, 16 * v:16 * (v + 1)]                 # [B, 48, S, 16, 16]
            x = self.stem(zv.permute(0, 2, 1, 3, 4).reshape(b * s, 48, 16, 16))  # [B*S, dim, 8, 8]
            x = x.flatten(2).transpose(1, 2).reshape(b, s, 64, self.dim)
            x = x + self.pos_emb + self.slot_emb[self.slots][None, :, None] + self.view_emb[v]
            out.append(x.reshape(b, s * 64, self.dim))
        return torch.cat(out, 1)

    def forward(self, z, action, task_id, swap_future=False):
        """Returns cost logits [B, 8, K] and proximity logits [B, 8, 2]."""
        tokens = [self.act((action - self.act_mu) / self.act_sd) + self.step_emb, self.task_emb(task_id)[:, None]]
        if self.slots:
            if swap_future and 2 in self.slots:  # dependency diagnostic: futures of other samples
                z = torch.cat([z[:, :, :2], z[:, :, 2:].roll(1, dims=0)], 2)
            tokens.append(self.visual_tokens(z))
        h = self.norm(self.encoder(torch.cat(tokens, 1)))[:, :CHUNK]
        return self.cost_head(h), self.prox_head(h)
