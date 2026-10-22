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

"""Dual-view (agentview + wrist) Wan without touching the reference tree.

A dual frame is the two 256x256 views stacked vertically (agent on top, wrist
below) into one 512x256 image, so the pipeline, sampler and trainer keep their
single-image interfaces. Two patches make this a latent-level concatenation:

* VAE: a 2:1 (H = 2W) input is split into its halves, each half is encoded on its
  own and the latents are concatenated along H (16x16 + 16x16 -> 32x16). Decoding
  splits the latent the same way. The views never share a VAE convolution, so no
  seam artefacts and the VAE sees exactly the 256x256 inputs it was trained on.
* Action modulation: model_fn adds one action embedding per latent frame to the
  timestep embedding and repeats it over a hard-coded 64 tokens (16x16 latent with
  2x2 patches). A dual frame has 128 tokens, so action_mlp2's output is repeated
  ``views`` times per frame; tokens are frame-major, so frame k's 128 tokens all
  get frame k's embedding. The forward is patched on the instance, so parameter
  names (and checkpoint keys) are unchanged.

RoPE and the per-token timestep already follow the latent shape. In the DiT the
wrist tokens sit at rows 16-31 of the same frame, so attention spans both views.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

_VAE_PATCHED = False


def stack_views(agent: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """[..., H, W, 3] x 2 -> [..., 2H, W, 3] (agent on top)."""
    return np.concatenate([agent, wrist], axis=-3)


def split_views(frames: np.ndarray):
    """Inverse of stack_views."""
    h = frames.shape[-3] // 2
    return frames[..., :h, :, :], frames[..., h:, :, :]


def stack_pil(agent_imgs, wrist_imgs):
    return [Image.fromarray(stack_views(np.asarray(a), np.asarray(w))) for a, w in zip(agent_imgs, wrist_imgs)]


def patch_vae_dual() -> None:
    global _VAE_PATCHED
    if _VAE_PATCHED:
        return
    from diffsynth.models import wan_video_vae as wv

    for cls in {c for c in vars(wv).values() if isinstance(c, type) and hasattr(c, "single_encode")}:
        enc, dec = cls.single_encode, cls.single_decode

        def single_encode(self, video, device, _enc=enc):
            h, w = video.shape[-2:]
            if h == 2 * w:
                top, bottom = video[..., : h // 2, :], video[..., h // 2:, :]
                return torch.cat([_enc(self, top, device), _enc(self, bottom, device)], dim=-2)
            return _enc(self, video, device)

        def single_decode(self, hidden_state, device, _dec=dec):
            h, w = hidden_state.shape[-2:]
            if h == 2 * w:
                top, bottom = hidden_state[..., : h // 2, :], hidden_state[..., h // 2:, :]
                return torch.cat([_dec(self, top, device), _dec(self, bottom, device)], dim=-2)
            return _dec(self, hidden_state, device)

        cls.single_encode, cls.single_decode = single_encode, single_decode
    _VAE_PATCHED = True


def patch_action_tokens(dit: torch.nn.Module, views: int = 2) -> None:
    """Repeat each latent frame's modulation embedding ``views`` times (see module doc)."""
    m = dit.action_mlp2
    if getattr(m, "_views", None) == views:
        return
    orig = type(m).forward

    def forward(x, _m=m):
        y = orig(_m, x)
        return y.repeat_interleave(_m._views, dim=-2) if _m._views > 1 else y

    m._views = views
    m.forward = forward


def enable_dual(pipe, views: int = 2) -> None:
    patch_vae_dual()
    patch_action_tokens(pipe.dit, views)
