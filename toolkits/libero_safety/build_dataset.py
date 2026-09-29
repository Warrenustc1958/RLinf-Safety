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

"""Build aligned Wan trajectories and SAM labels from safety audit episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

AUDIT_SCHEMA_VERSION = "libero_safety_episode_audit/v1"
DATASET_SCHEMA_VERSION = "libero_safety_wan_sam_dataset/v1"
QUADRANTS = ("Q1", "Q2", "Q3", "Q4")
DEFAULT_RISK_HORIZONS = (1, 4, 8, 16, 32)


@dataclass(frozen=True)
class ValidatedEpisode:
    """One audit record with a verified, action-aligned visual sidecar."""

    audit_path: Path
    visual_path: Path
    record: dict[str, Any]
    collection_id: str
    source_episode_id: str
    episode_id: str
    quadrant: str
    num_transitions: int
    camera_key: str


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def discover_audits(roots: Iterable[Path]) -> list[Path]:
    """Discover only auditor-v1 JSON objects below one or more roots."""
    results: list[Path] = []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"Audit root does not exist: {root}")
        candidates = [root] if root.is_file() else sorted(root.rglob("*.json"))
        for path in candidates:
            try:
                record = _load_json(path)
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            if record.get("schema_version") == AUDIT_SCHEMA_VERSION:
                results.append(path)
    return sorted(set(results))


def validate_episode(
    path: Path, camera_key: str, collection_id: str = "collection"
) -> ValidatedEpisode:
    """Validate audit/sidecar alignment without trusting filenames."""
    record = _load_json(path)
    if record.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported audit schema")
    episode = record.get("episode")
    trajectory = record.get("trajectory")
    outcome = record.get("outcome")
    if not isinstance(episode, dict) or not isinstance(outcome, dict):
        raise ValueError(f"{path}: missing episode or outcome object")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError(f"{path}: trajectory must be non-empty")
    source_episode_id = str(episode.get("episode_id", ""))
    if not source_episode_id:
        raise ValueError(f"{path}: missing episode_id")
    episode_id = f"{collection_id}:{source_episode_id}"
    quadrant = str(outcome.get("quadrant", record.get("quadrant", "")))
    if quadrant not in QUADRANTS:
        raise ValueError(f"{path}: invalid quadrant {quadrant!r}")
    expected_quadrant = {
        (True, False): "Q1",
        (True, True): "Q2",
        (False, False): "Q3",
        (False, True): "Q4",
    }[(bool(outcome.get("success")), bool(outcome.get("unsafe")))]
    if quadrant != expected_quadrant:
        raise ValueError(
            f"{path}: quadrant {quadrant} disagrees with success/unsafe "
            f"({expected_quadrant})"
        )
    visual = record.get("visual_trajectory")
    if not isinstance(visual, dict):
        raise ValueError(
            f"{path}: no visual_trajectory; collect with "
            "episode_auditor.save_visual_observations=true"
        )
    relative_visual_path = Path(str(visual.get("path", "")))
    if relative_visual_path.is_absolute() or ".." in relative_visual_path.parts:
        raise ValueError(f"{path}: visual sidecar path must be local and relative")
    visual_path = path.parent / relative_visual_path
    if not visual_path.is_file():
        raise ValueError(f"{path}: visual sidecar not found: {visual_path}")
    expected_frames = len(trajectory) + 1
    if int(visual.get("num_frames", -1)) != expected_frames:
        raise ValueError(
            f"{path}: visual metadata has {visual.get('num_frames')} frames; "
            f"expected {expected_frames}"
        )
    with np.load(visual_path, allow_pickle=False) as sidecar:
        if camera_key not in sidecar.files:
            raise ValueError(
                f"{path}: camera {camera_key!r} not in sidecar {sidecar.files}"
            )
        images = sidecar[camera_key]
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"{visual_path}: expected [T+1,H,W,3], got {images.shape}")
        if len(images) != expected_frames:
            raise ValueError(
                f"{visual_path}: got {len(images)} frames; expected {expected_frames}"
            )
    for index, step in enumerate(trajectory):
        if not isinstance(step, dict):
            raise ValueError(f"{path}: trajectory[{index}] is not an object")
        if "action" not in step or "predicates" not in step:
            raise ValueError(f"{path}: trajectory[{index}] misses action/predicates")
    return ValidatedEpisode(
        audit_path=path,
        visual_path=visual_path,
        record=record,
        collection_id=collection_id,
        source_episode_id=source_episode_id,
        episode_id=episode_id,
        quadrant=quadrant,
        num_transitions=len(trajectory),
        camera_key=camera_key,
    )


def _parse_quadrant_counts(values: list[str], option: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for value in values:
        try:
            quadrant, raw_count = value.split("=", maxsplit=1)
            count = int(raw_count)
        except ValueError as error:
            raise ValueError(f"{option} expects Q1=COUNT, got {value!r}") from error
        quadrant = quadrant.upper()
        if quadrant not in QUADRANTS or count < 0:
            raise ValueError(f"{option} has invalid value {value!r}")
        parsed[quadrant] = count
    return parsed


def _stable_score(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _split_group(episode: ValidatedEpisode, split_unit: str) -> str:
    metadata = episode.record["episode"]
    if split_unit == "task":
        return ":".join(
            [
                str(metadata.get("task_suite")),
                str(metadata.get("safety_level")),
                str(metadata.get("task_id")),
            ]
        )
    if split_unit == "task_trial":
        return ":".join(
            [
                str(metadata.get("task_suite")),
                str(metadata.get("safety_level")),
                str(metadata.get("task_id")),
                str(metadata.get("trial_id")),
            ]
        )
    source_root_id = episode.record.get("root_episode_id")
    if source_root_id is None:
        return episode.episode_id
    return f"{episode.collection_id}:{source_root_id}"


def _collection_id(root: Path) -> str:
    """Create a stable namespace because auditor IDs are local to one run."""
    resolved = str(root.resolve())
    return f"collection-{hashlib.sha256(resolved.encode()).hexdigest()[:12]}"


def _audit_collection_root(path: Path) -> Path:
    """Find the auditor root even when the CLI root contains many runs."""
    for parent in path.parents:
        if parent.name.startswith("process_"):
            return parent.parent
    return path.parent


def _assign_split(
    episode: ValidatedEpisode,
    split_unit: str,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> str:
    unit = _split_group(episode, split_unit)
    ratio = _stable_score(seed, unit) / float(2**64)
    if ratio < train_fraction:
        return "train"
    if ratio < train_fraction + val_fraction:
        return "val"
    return "test"


def _quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Convert LIBERO xyzw quaternion to the same axis-angle used by its env."""
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {quat.shape}")
    if quat[3] > 1.0:
        quat[3] = 1.0
    denominator = np.sqrt(max(1.0 - quat[3] * quat[3], 0.0))
    if denominator < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * np.arccos(quat[3]) / denominator)).astype(np.float32)


def _initial_state(record: dict[str, Any]) -> np.ndarray:
    observation = record.get("initial_observation")
    if not isinstance(observation, dict):
        raise ValueError("Audit must include initial_observation for Wan init state")
    try:
        position = np.asarray(observation["robot0_eef_pos"], dtype=np.float32)
        axis_angle = _quat_to_axis_angle(
            np.asarray(observation["robot0_eef_quat"], dtype=np.float32)
        )
        gripper = np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32)
    except KeyError as error:
        raise ValueError(
            f"Initial observation misses state key {error.args[0]!r}"
        ) from error
    return np.concatenate([position, axis_angle, gripper]).astype(np.float32)


def _predicate_definitions(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    definitions: dict[str, dict[str, Any]] = {}
    for step in trajectory:
        for predicate in step["predicates"]:
            instance_id = str(predicate["instance_id"])
            definition = {
                key: predicate[key]
                for key in (
                    "instance_id",
                    "source",
                    "predicate",
                    "arguments",
                    "semantics",
                )
            }
            previous = definitions.setdefault(instance_id, definition)
            if previous != definition:
                raise ValueError(f"Predicate definition changed for {instance_id}")
    return [definitions[key] for key in sorted(definitions)]


def _step_labels(
    episode: ValidatedEpisode, risk_horizons: tuple[int, ...]
) -> list[dict[str, Any]]:
    trajectory = episode.record["trajectory"]
    violation_indices = [
        index
        for index, step in enumerate(trajectory)
        if bool(step["terminations"]["safety_violation"])
    ]
    labels: list[dict[str, Any]] = []
    unsafe_so_far = False
    for index, step in enumerate(trajectory):
        current_violation = bool(step["terminations"]["safety_violation"])
        unsafe_so_far |= current_violation
        future = next((item for item in violation_indices if item >= index), None)
        time_to_violation = None if future is None else future - index
        active_constraints = [
            str(predicate["instance_id"])
            for predicate in step["predicates"]
            if predicate["source"] == "constraint" and bool(predicate["value"])
        ]
        labels.append(
            {
                "episode_id": episode.episode_id,
                "timestep": int(step["timestep"]),
                "action_index": index,
                "label_alignment": "post_action_state",
                "executed_action": step["action"],
                "episode_success": bool(episode.record["outcome"]["success"]),
                "episode_unsafe": bool(episode.record["outcome"]["unsafe"]),
                "quadrant": episode.quadrant,
                "post_action_violation": current_violation,
                "post_action_severe_collision": bool(
                    step["terminations"]["severe_collision"]
                ),
                "unsafe_so_far": unsafe_so_far,
                "active_constraint_instance_ids": active_constraints,
                "predicate_values": {
                    str(predicate["instance_id"]): bool(predicate["value"])
                    for predicate in step["predicates"]
                },
                "time_to_next_violation_steps": time_to_violation,
                "violation_within_horizon": {
                    str(horizon): time_to_violation is not None
                    and time_to_violation < horizon
                    for horizon in risk_horizons
                },
                "severity": (
                    "severe"
                    if step["terminations"]["severe_collision"]
                    else "violation"
                    if current_violation
                    else "safe"
                ),
                "label_source": "libero_safety_simulator_predicates",
            }
        )
    return labels


def _wan_frames(episode: ValidatedEpisode, labels: list[dict[str, Any]]) -> np.ndarray:
    trajectory = episode.record["trajectory"]
    instruction = str(episode.record["episode"]["instruction"])
    init_state = _initial_state(episode.record)
    with np.load(episode.visual_path, allow_pickle=False) as sidecar:
        images = sidecar[episode.camera_key]
        frames = []
        for index, (step, label) in enumerate(zip(trajectory, labels)):
            frames.append(
                {
                    "image": np.asarray(images[index], dtype=np.uint8),
                    "delta_action": np.asarray(step["action"], dtype=np.float32),
                    "init_ee_pose": init_state,
                    "instruction": instruction,
                    "episode_id": episode.episode_id,
                    "timestep": int(step["timestep"]),
                    "quadrant": episode.quadrant,
                    "success": bool(episode.record["outcome"]["success"]),
                    "unsafe": bool(episode.record["outcome"]["unsafe"]),
                    "post_action_violation": label["post_action_violation"],
                    "post_action_severe_collision": label[
                        "post_action_severe_collision"
                    ],
                }
            )
    return np.asarray(frames, dtype=object)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")
    temporary.replace(path)


def build_dataset(
    *,
    audit_roots: list[Path],
    output_dir: Path,
    quadrants: tuple[str, ...] = QUADRANTS,
    camera_key: str = "agentview_image",
    split_unit: str = "root",
    train_fraction: float = 0.9,
    val_fraction: float = 0.1,
    seed: int = 0,
    risk_horizons: tuple[int, ...] = DEFAULT_RISK_HORIZONS,
    max_per_quadrant: dict[str, int] | None = None,
    min_per_quadrant: dict[str, int] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate, select, split, and export one unified Wan/SAM dataset."""
    if train_fraction < 0 or val_fraction < 0 or train_fraction + val_fraction > 1:
        raise ValueError("train/val fractions must be non-negative and sum to <= 1")
    if split_unit not in {"root", "task", "task_trial"}:
        raise ValueError("split_unit must be root, task, or task_trial")
    selected_quadrants = tuple(dict.fromkeys(item.upper() for item in quadrants))
    if not selected_quadrants or any(
        item not in QUADRANTS for item in selected_quadrants
    ):
        raise ValueError(f"quadrants must be selected from {QUADRANTS}")

    unique_roots: list[Path] = []
    seen_roots: set[Path] = set()
    for root in audit_roots:
        resolved = root.resolve()
        if resolved not in seen_roots:
            seen_roots.add(resolved)
            unique_roots.append(root)
    audit_sources = [
        (_collection_id(_audit_collection_root(path)), path)
        for root in unique_roots
        for path in discover_audits([root])
    ]
    audit_paths = [path for _, path in audit_sources]
    if not audit_paths:
        raise ValueError(f"No {AUDIT_SCHEMA_VERSION} JSON files found")
    valid: list[ValidatedEpisode] = []
    invalid: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for collection_id, path in audit_sources:
        try:
            episode = validate_episode(path, camera_key, collection_id)
            if episode.episode_id in seen_ids:
                raise ValueError(f"duplicate episode_id {episode.episode_id}")
            seen_ids.add(episode.episode_id)
            valid.append(episode)
        except (KeyError, TypeError, ValueError) as error:
            invalid.append({"path": str(path), "error": str(error)})
    if invalid:
        examples = "; ".join(item["error"] for item in invalid[:3])
        raise ValueError(
            f"Refusing to build with {len(invalid)} invalid audit episode(s): {examples}"
        )

    eligible = [episode for episode in valid if episode.quadrant in selected_quadrants]
    grouped: dict[str, list[ValidatedEpisode]] = defaultdict(list)
    for episode in eligible:
        grouped[episode.quadrant].append(episode)
    selected: list[ValidatedEpisode] = []
    max_per_quadrant = max_per_quadrant or {}
    for quadrant in selected_quadrants:
        ordered = sorted(
            grouped[quadrant],
            key=lambda item: (_stable_score(seed, item.episode_id), item.episode_id),
        )
        limit = max_per_quadrant.get(quadrant)
        selected.extend(ordered if limit is None else ordered[:limit])
    selected.sort(key=lambda item: item.episode_id)
    selected_counts = Counter(item.quadrant for item in selected)
    min_per_quadrant = min_per_quadrant or {}
    shortfalls = {
        quadrant: minimum - selected_counts[quadrant]
        for quadrant, minimum in min_per_quadrant.items()
        if selected_counts[quadrant] < minimum
    }
    if shortfalls:
        raise ValueError(f"Quadrant minimums are not met; shortfalls={shortfalls}")
    if not selected:
        raise ValueError("No episodes remain after quadrant selection")

    split_by_id = {
        item.episode_id: _assign_split(
            item, split_unit, train_fraction, val_fraction, seed
        )
        for item in selected
    }
    summary: dict[str, Any] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "audit_roots": [str(path.resolve()) for path in unique_roots],
        "camera_key": camera_key,
        "quadrants": list(selected_quadrants),
        "episodes_discovered": len(audit_paths),
        "episodes_selected": len(selected),
        "transitions_selected": sum(item.num_transitions for item in selected),
        "quadrant_counts": {
            quadrant: selected_counts[quadrant] for quadrant in QUADRANTS
        },
        "split_counts": dict(Counter(split_by_id.values())),
        "split_unit": split_unit,
        "risk_horizons": list(risk_horizons),
        "complete_quadrant_coverage": all(
            selected_counts[item] > 0 for item in QUADRANTS
        ),
        "ready_for_wan": True,
        "ready_for_sam": True,
        "dry_run": dry_run,
    }
    if dry_run:
        return summary
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory must be absent or empty to avoid mixing datasets: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sam").mkdir()
    for split in ("train", "val", "test"):
        (output_dir / "wan" / split).mkdir(parents=True)

    episode_manifest: list[dict[str, Any]] = []
    sam_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    split_indices = Counter()
    for episode in selected:
        split = split_by_id[episode.episode_id]
        labels = _step_labels(episode, risk_horizons)
        wan_frames = _wan_frames(episode, labels)
        wan_name = f"traj{split_indices[split]:06d}.npy"
        split_indices[split] += 1
        wan_path = output_dir / "wan" / split / wan_name
        temporary_wan_path = wan_path.with_suffix(".npy.tmp")
        with temporary_wan_path.open("wb") as file:
            np.save(file, wan_frames, allow_pickle=True)
        temporary_wan_path.replace(wan_path)
        for label in labels:
            label["split"] = split
            label["wan_path"] = str(wan_path.relative_to(output_dir))
        sam_rows[split].extend(labels)
        metadata = episode.record["episode"]
        source_root_id = episode.record.get("root_episode_id")
        source_branch_id = episode.record.get("branch_id")
        episode_manifest.append(
            {
                "episode_id": episode.episode_id,
                "source_episode_id": episode.source_episode_id,
                "collection_id": episode.collection_id,
                "root_episode_id": (
                    f"{episode.collection_id}:{source_root_id}"
                    if source_root_id is not None
                    else None
                ),
                "branch_id": (
                    f"{episode.collection_id}:{source_branch_id}"
                    if source_branch_id is not None
                    else None
                ),
                "alpha": episode.record.get("alpha"),
                "perturbation_range": episode.record.get("perturbation_range"),
                "branch_timestep": metadata.get("branch_timestep"),
                "episode_kind": metadata.get("episode_kind", "root"),
                "split": split,
                "quadrant": episode.quadrant,
                "success": bool(episode.record["outcome"]["success"]),
                "unsafe": bool(episode.record["outcome"]["unsafe"]),
                "task_suite": metadata.get("task_suite"),
                "safety_level": metadata.get("safety_level"),
                "task_id": metadata.get("task_id"),
                "trial_id": metadata.get("trial_id"),
                "instruction": metadata.get("instruction"),
                "num_transitions": episode.num_transitions,
                "first_violation_timestep": episode.record["outcome"].get(
                    "first_violation_timestep"
                ),
                "first_severe_collision_timestep": episode.record["outcome"].get(
                    "first_severe_collision_timestep"
                ),
                "predicate_instances": _predicate_definitions(
                    episode.record["trajectory"]
                ),
                "audit_path": str(episode.audit_path.resolve()),
                "visual_path": str(episode.visual_path.resolve()),
                "wan_path": str(wan_path.relative_to(output_dir)),
                "audit_sha256": _sha256(episode.audit_path),
                "visual_sha256": _sha256(episode.visual_path),
            }
        )
    _write_jsonl(output_dir / "manifest.jsonl", episode_manifest)
    for split in ("train", "val", "test"):
        _write_jsonl(output_dir / "sam" / f"{split}.jsonl", sam_rows[split])
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build aligned Wan .npy trajectories and SAM JSONL labels."
    )
    parser.add_argument("--audit-root", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quadrants", nargs="+", default=list(QUADRANTS))
    parser.add_argument("--camera-key", default="agentview_image")
    parser.add_argument(
        "--split-unit", choices=("root", "task", "task_trial"), default="root"
    )
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--risk-horizons", nargs="+", type=int, default=list(DEFAULT_RISK_HORIZONS)
    )
    parser.add_argument("--max-per-quadrant", action="append", default=[])
    parser.add_argument("--min-per-quadrant", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = build_dataset(
        audit_roots=args.audit_root,
        output_dir=args.output_dir,
        quadrants=tuple(args.quadrants),
        camera_key=args.camera_key,
        split_unit=args.split_unit,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
        risk_horizons=tuple(args.risk_horizons),
        max_per_quadrant=_parse_quadrant_counts(
            args.max_per_quadrant, "--max-per-quadrant"
        ),
        min_per_quadrant=_parse_quadrant_counts(
            args.min_per_quadrant, "--min-per-quadrant"
        ),
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
