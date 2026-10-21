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

"""fp32 master weights for the Wan DiT without touching the reference tree.

The pipeline loads the DiT in bf16 and the reference trainer steps AdamW on those
bf16 tensors. bf16 spacing is about |w| / 256, while an Adam step is about lr
(1e-5), so most updates round away (87 % of elements unchanged on a measured
tensor). Keeping the trainable weights in fp32 and running the forward under bf16
autocast fixes that. Two places need help under fp32 weights:

* ``flash_attention`` receives fp32 q / k because RMSNorm multiplies by an fp32
  weight; flash-attn only accepts fp16 / bf16, so inputs are cast to bf16 and the
  output back to the caller's dtype.
"""

from __future__ import annotations

import torch

_PATCHED = False


def patch_flash_attention(dtype: torch.dtype = torch.bfloat16) -> None:
    global _PATCHED
    if _PATCHED:
        return
    from diffsynth.models import wan_video_dit as wvd

    orig = wvd.flash_attention

    def flash_attention(q, k, v, num_heads, compatibility_mode=False):
        out_dtype = v.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
        return orig(q, k, v, num_heads=num_heads, compatibility_mode=compatibility_mode).to(out_dtype)

    wvd.flash_attention = flash_attention
    _PATCHED = True


def to_fp32_master(module: torch.nn.Module) -> int:
    """Cast trainable parameters to fp32; frozen ones (VAE) stay as loaded."""
    n = 0
    for p in module.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
            n += 1
    return n
