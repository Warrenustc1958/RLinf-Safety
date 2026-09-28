# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""LeRobot-v2 SFT input pipeline for OpenVLA-OFT.

The loader intentionally reads local parquet and MP4 files directly.  This
keeps large, partially downloaded datasets usable and avoids importing the
TensorFlow/RLDS stack used by the upstream OpenVLA-OFT training script.
"""

from __future__ import annotations

import bisect
import functools
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader

IGNORE_INDEX = -100
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Episode:
    index: int
    length: int
    task: str


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def openvla_action_stats(dataset_root: str | Path) -> dict[str, Any]:
    """Convert LeRobot statistics to OpenVLA's q01/q99 schema."""
    root = Path(dataset_root)
    stats = _read_json(root / "meta" / "stats.json")
    action = stats["actions"]
    action_dim = len(action["mean"])
    mask = [True] * action_dim
    # LIBERO's last dimension is an already-normalized binary gripper command.
    if action_dim:
        mask[-1] = False
    return {
        "action": {**action, "mask": mask},
        "proprio": _read_json(root / "meta" / "stats.json")["observation.state"],
    }


def discover_openvla_episodes(
    dataset_root: str | Path,
    *,
    video_keys: Sequence[str],
    action_horizon: int,
    require_complete_videos: bool = True,
    max_episodes: int | None = None,
) -> tuple[list[_Episode], int]:
    """Return trainable episodes and the total episode count in metadata."""
    root = Path(dataset_root)
    info = _read_json(root / "meta" / "info.json")
    rows = _read_jsonl(root / "meta" / "episodes.jsonl")
    selected: list[_Episode] = []
    for row in rows:
        episode_index = int(row["episode_index"])
        length = int(row["length"])
        if length < action_horizon:
            continue
        if require_complete_videos:
            chunk = episode_index // int(info["chunks_size"])
            paths = [
                root
                / str(info["video_path"]).format(
                    episode_chunk=chunk,
                    video_key=key,
                    episode_index=episode_index,
                )
                for key in video_keys
            ]
            if not all(path.is_file() for path in paths):
                continue
        tasks = row.get("tasks") or []
        if not tasks:
            raise ValueError(f"Episode {episode_index} has no task description")
        selected.append(_Episode(episode_index, length, str(tasks[0])))
        if max_episodes is not None and len(selected) >= max_episodes:
            break
    return selected, int(info["total_episodes"])


class OpenVLAOFTLeRobotDataset(Dataset):
    """Map-style OpenVLA-OFT dataset over a local LeRobot-v2 tree."""

    def __init__(
        self,
        dataset_root: str | Path,
        tokenizer: Any,
        image_processor: Any,
        *,
        action_horizon: int = 8,
        video_keys: Sequence[str] = ("observation.image",),
        require_complete_videos: bool = True,
        max_episodes: int | None = None,
        parquet_cache_size: int = 8,
        video_cache_size: int = 4,
    ) -> None:
        self.root = Path(dataset_root).expanduser().resolve()
        self.info = _read_json(self.root / "meta" / "info.json")
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.action_horizon = int(action_horizon)
        self.video_keys = tuple(video_keys)
        self.parquet_cache_size = int(parquet_cache_size)
        self.video_cache_size = int(video_cache_size)
        self._parquet_cache: OrderedDict[int, Any] = OrderedDict()
        self._video_cache: OrderedDict[tuple[int, str], Any] = OrderedDict()

        self.episodes, total_episodes = discover_openvla_episodes(
            self.root,
            video_keys=self.video_keys,
            action_horizon=self.action_horizon,
            require_complete_videos=require_complete_videos,
            max_episodes=max_episodes,
        )
        if not self.episodes:
            raise RuntimeError(f"No trainable episodes found under {self.root}")

        counts = [ep.length - self.action_horizon + 1 for ep in self.episodes]
        self._ends = np.cumsum(counts, dtype=np.int64).tolist()
        self.stats = openvla_action_stats(self.root)
        self._action_low = np.asarray(self.stats["action"]["q01"], dtype=np.float32)
        self._action_high = np.asarray(self.stats["action"]["q99"], dtype=np.float32)
        self._action_mask = np.asarray(self.stats["action"]["mask"], dtype=bool)
        logger.info(
            "OpenVLA-OFT LeRobot dataset: %d/%d episodes, %d samples, videos=%s",
            len(self.episodes),
            total_episodes,
            len(self),
            self.video_keys,
        )

    @property
    def coverage(self) -> tuple[int, int]:
        return len(self.episodes), int(self.info["total_episodes"])

    def __len__(self) -> int:
        return int(self._ends[-1])

    def _index(self, index: int) -> tuple[_Episode, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        ep_pos = bisect.bisect_right(self._ends, index)
        start = 0 if ep_pos == 0 else self._ends[ep_pos - 1]
        return self.episodes[ep_pos], index - start

    def _episode_table(self, episode_index: int):
        table = self._parquet_cache.pop(episode_index, None)
        if table is None:
            chunk = episode_index // int(self.info["chunks_size"])
            path = self.root / str(self.info["data_path"]).format(
                episode_chunk=chunk, episode_index=episode_index
            )
            table = pq.read_table(path, columns=["actions"])
        self._parquet_cache[episode_index] = table
        while len(self._parquet_cache) > self.parquet_cache_size:
            self._parquet_cache.popitem(last=False)
        return table

    def _video_path(self, episode_index: int, video_key: str) -> Path:
        chunk = episode_index // int(self.info["chunks_size"])
        return self.root / str(self.info["video_path"]).format(
            episode_chunk=chunk,
            video_key=video_key,
            episode_index=episode_index,
        )

    def _decode_frame(self, episode_index: int, video_key: str, frame_index: int):
        from torchcodec.decoders import VideoDecoder

        cache_key = (episode_index, video_key)
        decoder = self._video_cache.pop(cache_key, None)
        if decoder is None:
            decoder = VideoDecoder(
                str(self._video_path(episode_index, video_key)),
                device="cpu",
                seek_mode="approximate",
                dimension_order="NCHW",
            )
        self._video_cache[cache_key] = decoder
        while len(self._video_cache) > self.video_cache_size:
            self._video_cache.popitem(last=False)
        return decoder.get_frames_at(indices=[frame_index]).data[0]

    def _normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        normalized = np.where(
            self._action_mask,
            2
            * (actions - self._action_low)
            / (self._action_high - self._action_low + 1e-8)
            - 1,
            actions,
        )
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def _action_token_ids(self, actions: np.ndarray) -> list[int]:
        bins = np.linspace(-1.0, 1.0, 256)
        discrete = np.digitize(actions, bins)
        return (int(self.tokenizer.vocab_size) - discrete).reshape(-1).tolist()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode, frame_index = self._index(index)
        table = self._episode_table(episode.index)
        actions = np.asarray(
            table["actions"].slice(frame_index, self.action_horizon).to_pylist(),
            dtype=np.float32,
        )
        actions = self._normalize_actions(actions)

        prompt = (
            f"In: What action should the robot take to {episode.task.lower()}?\nOut: "
        )
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True).input_ids
        action_ids = self._action_token_ids(actions)
        input_ids = torch.tensor(
            prompt_ids + action_ids + [self.tokenizer.eos_token_id], dtype=torch.long
        )
        labels = input_ids.clone()
        labels[: len(prompt_ids)] = IGNORE_INDEX

        frames = [
            self._decode_frame(episode.index, key, frame_index)
            for key in self.video_keys
        ]
        # Each fused DINOv2/SigLIP image contributes six channels after transform.
        pixels = []
        for frame in frames:
            pixel = self.image_processor(
                frame.unsqueeze(0).unsqueeze(0), return_tensors="pt"
            )["pixel_values"].squeeze(0)
            pixels.append(pixel.reshape(-1, *pixel.shape[-2:]))
        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": torch.cat(pixels, dim=0),
        }


def _collate_openvla_oft(
    samples: Sequence[dict[str, torch.Tensor]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    input_ids = pad_sequence(
        [sample["input_ids"] for sample in samples],
        batch_first=True,
        padding_value=pad_token_id,
    )
    labels = pad_sequence(
        [sample["labels"] for sample in samples],
        batch_first=True,
        padding_value=IGNORE_INDEX,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": input_ids.ne(pad_token_id),
        "labels": labels,
        "pixel_values": torch.stack([sample["pixel_values"] for sample in samples]),
    }


class _EpisodeGroupedDistributedSampler(Sampler[int]):
    """Shuffle episodes and frames while retaining decoder locality per rank."""

    def __init__(
        self,
        dataset: OpenVLAOFTLeRobotDataset,
        *,
        num_replicas: int,
        rank: int,
        seed: int,
        shuffle: bool,
        drop_last: bool,
    ) -> None:
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        if self.drop_last:
            self.num_samples = len(dataset) // self.num_replicas
        else:
            self.num_samples = (
                len(dataset) + self.num_replicas - 1
            ) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        episode_order = list(range(len(self.dataset.episodes)))
        if self.shuffle:
            episode_order = torch.randperm(
                len(episode_order), generator=generator
            ).tolist()

        starts = [0] + self.dataset._ends[:-1]
        indices: list[int] = []
        for episode_pos in episode_order:
            episode_indices = list(
                range(starts[episode_pos], self.dataset._ends[episode_pos])
            )
            if self.shuffle:
                order = torch.randperm(
                    len(episode_indices), generator=generator
                ).tolist()
                episode_indices = [episode_indices[i] for i in order]
            indices.extend(episode_indices)

        if self.drop_last:
            indices = indices[: self.total_size]
        elif len(indices) < self.total_size:
            indices.extend(indices[: self.total_size - len(indices)])
        offset = self.rank * self.num_samples
        return iter(indices[offset : offset + self.num_samples])


def build_openvla_oft_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
):
    """Build the distributed OpenVLA-OFT SFT loader."""
    if not isinstance(data_paths, str):
        paths = list(data_paths)
        if len(paths) != 1:
            raise ValueError("OpenVLA-OFT SFT currently accepts one LeRobot root")
        data_paths = str(paths[0])

    from rlinf.models.embodiment.openvla_oft.rlinf import (
        get_model_config_and_input_processor,
    )

    _, processor = get_model_config_and_input_processor(cfg.actor.model)
    data_cfg = cfg.data
    if cfg.actor.model.get("implement_version", "rlinf") != "rlinf":
        raise NotImplementedError(
            "The LeRobot OpenVLA-OFT SFT loader currently supports "
            "implement_version=rlinf only"
        )
    video_keys = data_cfg.get("video_keys", ["observation.image"])
    dataset = OpenVLAOFTLeRobotDataset(
        data_paths,
        processor.tokenizer,
        processor.image_processor,
        action_horizon=int(cfg.actor.model.num_action_chunks),
        video_keys=list(video_keys),
        require_complete_videos=bool(data_cfg.get("require_complete_videos", True)),
        max_episodes=data_cfg.get("max_episodes"),
        parquet_cache_size=int(data_cfg.get("parquet_cache_size", 8)),
        video_cache_size=int(data_cfg.get("video_cache_size", 4)),
    )
    sampler = _EpisodeGroupedDistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        seed=int(cfg.actor.get("seed", 42)),
        shuffle=not eval_dataset,
        drop_last=not eval_dataset,
    )
    loader = StatefulDataLoader(
        dataset,
        batch_size=int(cfg.actor.micro_batch_size),
        sampler=sampler,
        num_workers=int(data_cfg.get("num_workers", 2)),
        collate_fn=functools.partial(
            _collate_openvla_oft, pad_token_id=processor.tokenizer.pad_token_id
        ),
        pin_memory=True,
        drop_last=not eval_dataset,
    )
    covered, total = dataset.coverage
    return loader, {
        "num_samples": len(dataset),
        "covered_episodes": covered,
        "total_episodes": total,
    }
