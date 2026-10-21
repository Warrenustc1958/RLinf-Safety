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

"""Wan AC-SFT training entry over the S0 index.

Reuses ``WanTrainingModule`` and ``launch_training_task`` from diffsynth-studio's
``train_rlinf.py`` unchanged (imported from its path, the reference tree stays
read-only) and swaps ``RLinfNpyDataset`` for ``WanACSFTDataset``. Per-step wall
time and peak CUDA memory are written to ``<output_path>/step_stats.jsonl``.

``--bench-loader N`` only times N dataset items through a DataLoader and exits.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import torch

from toolkits.libero_safety.wan_ac_sft.sampler import WanACSFTDataset

DIFFSYNTH_ROOT = os.environ.get("DIFFSYNTH_ROOT", "/mnt/nfs/fjj/Code/related_works/diffsynth-studio")


def _load_reference_trainer():
    path = Path(DIFFSYNTH_ROOT) / "examples/wanvideo/model_training/train_rlinf.py"
    spec = importlib.util.spec_from_file_location("diffsynth_train_rlinf", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TrainerDataset(torch.utils.data.Dataset):
    """Adapts ``WanACSFTDataset`` items to the reference trainer's expectations."""

    load_from_cache = False

    def __init__(self, inner: WanACSFTDataset):
        self.inner = inner

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        item = self.inner[i]
        return {"video": item["video"], "reference_image": item["reference_image"],
                "action": torch.from_numpy(item["action"])}


def main():
    ref = _load_reference_trainer()
    parser = ref.wan_parser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--view", default="agent", choices=["agent", "dual"])
    parser.add_argument("--train_length", type=int, default=100_000)
    parser.add_argument("--val_length", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context_noise_sigma", type=float, default=0.0)
    parser.add_argument("--static_video_prob", type=float, default=0.05)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--bench-loader", type=int, default=0)
    parser.add_argument("--keep_last", type=int, default=1, help="keep only the newest N step ckpts (0 = all)")
    parser.add_argument("--action_lr", type=float, default=None, help="separate lr for action_mlp* (default: same)")
    args = parser.parse_args()

    train = TrainerDataset(WanACSFTDataset(args.index, "train", args.view, args.train_length, args.seed))
    val = TrainerDataset(WanACSFTDataset(args.index, "val", args.view, args.val_length, args.seed + 1))

    if args.bench_loader:
        loader = torch.utils.data.DataLoader(train, shuffle=True, collate_fn=lambda x: x[0],
                                             num_workers=args.dataset_num_workers)
        t0, n = time.perf_counter(), 0
        for _ in loader:
            n += 1
            if n == 10:
                t_warm = time.perf_counter()
            if n >= args.bench_loader:
                break
        dt = time.perf_counter() - t_warm
        print(json.dumps({"bench_items": n - 10, "workers": args.dataset_num_workers,
                          "items_per_s": (n - 10) / dt, "s_per_item": dt / (n - 10)}))
        return

    model = ref.WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        audio_processor_config=args.audio_processor_config,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        context_noise_sigma=args.context_noise_sigma,
        static_video_prob=args.static_video_prob,
    )

    if args.action_lr is not None:
        # Separate learning rate for the action-injection MLPs via optimizer param
        # groups; the reference trainer passes trainable_modules() to AdamW.
        act = [p for n, p in model.named_parameters() if p.requires_grad and "action_mlp" in n]
        act_ids = {id(p) for p in act}
        rest = [p for p in model.parameters() if p.requires_grad and id(p) not in act_ids]
        if not act:
            raise ValueError("--action_lr given but no trainable action_mlp parameters found")
        model.trainable_modules = lambda: [{"params": rest}, {"params": act, "lr": args.action_lr}]
        print(f"[action_lr] {len(act)} action_mlp tensors at lr {args.action_lr}, "
              f"{len(rest)} other tensors at lr {args.learning_rate}", flush=True)

    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)
    stats = (out / "step_stats.jsonl").open("w")

    class TimedLogger(ref.ModelLogger):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._t = time.perf_counter()

        def on_step_end(self, accelerator, model, save_steps=None):
            super().on_step_end(accelerator, model, save_steps)
            now = time.perf_counter()
            if accelerator.is_main_process:
                stats.write(json.dumps({"step": self.num_steps, "sec": now - self._t,
                                        "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}) + "\n")
                stats.flush()
            self._t = now

        def save_model(self, accelerator, model, file_name):
            super().save_model(accelerator, model, file_name)
            if accelerator.is_main_process and args.keep_last > 0:
                ckpts = sorted(out.glob("step-*.safetensors"), key=lambda p: int(p.stem.split("-")[1]))
                for old in ckpts[:-args.keep_last]:
                    old.unlink()

    logger = TimedLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt)
    ref.launch_training_task(train, val, model, logger, args=args)
    stats.close()


if __name__ == "__main__":
    main()
