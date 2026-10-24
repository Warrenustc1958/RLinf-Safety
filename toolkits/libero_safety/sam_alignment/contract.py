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

"""Step 0: write experiment_contract.yaml (frozen versions and hashes for SAM v1)."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import time
from pathlib import Path

import yaml

from toolkits.libero_safety.sam_alignment.schema import CHANNELS, CHUNK, INDEX_ROOT, INDEXES, OUT_ROOT

RLINF = "/mnt/nfs/fjj/Code/RLinf-Safety"
CWS = "/mnt/nfs/fjj/Code/CodeWorld-Safe"
DIFFSYNTH = "/mnt/nfs/fjj/Code/related_works/diffsynth-studio"


def sha256(path: str, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def git_state(repo: str) -> dict:
    def run(*cmd):
        return subprocess.run(["git", "-C", repo, *cmd], capture_output=True, text=True).stdout.strip()
    return {"path": repo, "commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": [l for l in run("status", "--porcelain", "-uno").splitlines() if l]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dit", default="/data_cf/wan_ac_sft/runs/s3_dual_b10/ema-step-41000.safetensors")
    ap.add_argument("--vae", default="/mnt/nfs/fjj/wan_ac_sft/ckpt/RLinf-Wan-LIBERO-Spatial/Wan2.2_VAE.pth")
    ap.add_argument("--out", default=str(OUT_ROOT / "contract" / "experiment_contract.yaml"))
    args = ap.parse_args()

    t0 = time.time()
    contract = {
        "sam_version": "v1",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "channels": list(CHANNELS),
        "human_contact": "reported as a c_robot subset (hazard name contains 'hand'), no separate cost",
        "chunk_steps": CHUNK,
        "views": ["agentview", "wrist"],
        "dual_layout": "agent stacked over wrist (512x256); latent H rows 0:16 agent, 16:32 wrist",
        "latent_shapes": {"z_ctx": [48, 2, 32, 16], "z_future": [48, 2, 32, 16]},
        "window": "13 frames = anchor frame 0 + memory t0-3..t0 (clamped) + future t0+1..t0+8",
        "freeze_wan": True,
        "freeze_vae": True,
        "feature_source": "vae_latent",
        "force_supervision": False,
        "label_alignment": "post_action_state: cost[t] describes frame[t+1]",
        "label_authority": "CodeWorld-Safe safety_grader per-step grades[] in step_e audits",
        "action_space": {
            "sam": "audit action (executed env action, after pi0.5 unnormalize/output transform), "
                   "standardized with SAM cost_train statistics",
            "wan": "same action, pose dims x8, clip +-4, gripper unchanged (sampler.scale_pose), applied once",
        },
        "wan": {"dit": args.dit, "dit_sha256": sha256(args.dit), "vae": args.vae, "vae_sha256": sha256(args.vae),
                "selection": "dual s3_dual_b10 ema-step-41000, chosen by the user on 2026-10-24",
                "sampler": {"num_inference_steps": 5, "cfg_scale": 1.0, "tiled": False,
                            "seed_rule": "seed + 1000 * chunk (ar_eval.py)"}},
        "data": {name: {"manifest": str(INDEX_ROOT / name / "manifest.jsonl"),
                        "sha256": sha256(str(INDEX_ROOT / name / "manifest.jsonl"))} for name in INDEXES},
        "code": {"rlinf_safety": git_state(RLINF), "codeworld_safe": git_state(CWS),
                 "diffsynth": git_state(DIFFSYNTH)},
        "metrics": {
            "false_safe_units": {
                "step": "true violation at step t on channel k, predicted safe",
                "chunk": "Y_k = any valid violation in the chunk's 8 steps; predicted safe = max_t score < threshold",
                "event": "a contiguous run of violated steps; detected if any step of the run is flagged",
                "trajectory": "episode with any violation on channel k, never flagged",
            },
            "denominator": "only steps / chunks whose label is valid (valid_step & valid_channel); "
                           "chunk negatives require every step of the channel valid",
        },
        "not_in_v1": ["c_force head", "Wan backprop / guidance", "online ACI", "ROI latent-cosine h*",
                      "DiT features (Step 6.4 small probe only)"],
        "calibration_target": "to be fixed before Step 8 (per-channel miss rate vs coverage)",
        "data_usage_history": {
            "cws_v1 val/test": "used for Wan eval (eval_v2_dual) and RM rm_v1 checkpoint selection/test",
            "cws_test_new val/test": "used for Wan eval (eval_v2_dual_newtest) and RM rm_v1 selection/test",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(contract, sort_keys=False, allow_unicode=True))
    print(f"wrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
