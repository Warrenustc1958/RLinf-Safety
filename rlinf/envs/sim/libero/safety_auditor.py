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

"""Episode-level JSON auditing for LIBERO-Safety."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "libero_safety_episode_audit/v1"
AUDIT_MODES = ("official_eval", "shadow_collect")


def select_emitted_terminations(
    mode: str,
    raw_task_terminations: np.ndarray,
    official_terminations: np.ndarray,
    severe_collision_terminations: np.ndarray,
) -> np.ndarray:
    """Select the environment termination signal for an audit mode."""
    if mode == "official_eval":
        return np.asarray(official_terminations, dtype=bool).copy()
    if mode == "shadow_collect":
        return np.asarray(raw_task_terminations, dtype=bool) | np.asarray(
            severe_collision_terminations, dtype=bool
        )
    raise ValueError(
        f"Unsupported episode auditor mode {mode!r}; expected {AUDIT_MODES}"
    )


def detect_severe_collisions(
    predicate_audits: list[list[dict[str, Any]]],
    constraint_violations: np.ndarray,
    consecutive_violation_steps: np.ndarray,
    hard_stop_predicates: set[str],
    max_consecutive_violation_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect severe events from explicit predicates or persistent violations."""
    violations = np.asarray(constraint_violations, dtype=bool)
    streak = np.where(violations, consecutive_violation_steps + 1, 0).astype(np.int32)
    severe = np.zeros_like(violations)
    normalized_hard_stops = {name.lower() for name in hard_stop_predicates}
    for env_idx, predicates in enumerate(predicate_audits):
        severe[env_idx] = any(
            predicate.get("source") == "constraint"
            and bool(predicate.get("value"))
            and str(predicate.get("predicate", "")).lower() in normalized_hard_stops
            for predicate in predicates
        )
    severe |= streak >= max_consecutive_violation_steps
    return severe, streak


def _predicate_argument(value: Any) -> Any:
    """Normalize a parsed BDDL argument for subprocess transport."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_predicate_argument(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def audit_predicate_instances(env: Any) -> list[dict[str, Any]]:
    """Evaluate every parsed goal and constraint predicate independently."""
    task_env = env
    seen = set()
    while hasattr(task_env, "env") and id(task_env) not in seen:
        seen.add(id(task_env))
        nested = task_env.env
        if nested is task_env:
            break
        task_env = nested

    parsed_problem = getattr(task_env, "parsed_problem", None)
    eval_predicate = getattr(task_env, "_eval_predicate", None)
    if not isinstance(parsed_problem, dict) or not callable(eval_predicate):
        raise RuntimeError(
            "LIBERO-Safety auditing requires parsed_problem and _eval_predicate"
        )

    records = []
    predicate_groups = (
        ("goal", "goal_state", "satisfaction"),
        ("constraint", "constraints", "violation"),
    )
    for source, problem_key, semantics in predicate_groups:
        for index, instance in enumerate(parsed_problem.get(problem_key, [])):
            if not isinstance(instance, (list, tuple)) or not instance:
                raise RuntimeError(
                    f"Invalid {source} predicate instance at index {index}: "
                    f"{instance!r}"
                )
            records.append(
                {
                    "instance_id": f"{source}:{index:03d}",
                    "source": source,
                    "predicate": str(instance[0]),
                    "arguments": [
                        _predicate_argument(argument) for argument in instance[1:]
                    ],
                    "semantics": semantics,
                    "value": bool(eval_predicate(instance)),
                }
            )
    return records


def _json_value(value: Any) -> Any:
    """Convert simulator values to JSON-compatible Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy().tolist()
    return str(value)


def _non_visual_observation(observation: Any) -> Any:
    """Return JSON-safe observations without embedding image-sized arrays."""
    if not isinstance(observation, dict):
        return _json_value(observation)

    excluded_tokens = ("image", "depth", "segmentation")
    return {
        str(key): _json_value(value)
        for key, value in observation.items()
        if not any(token in str(key).lower() for token in excluded_tokens)
    }


class LiberoSafetyEpisodeAuditor:
    """Record complete LIBERO-Safety episode audit trails as atomic JSON files.

    ``official_eval`` finalizes on any safety violation. ``shadow_collect``
    records ordinary violations and continues until raw task success,
    truncation, or a severe-collision hard stop. Explicit reset and environment
    close also flush any non-empty trajectory in either mode.
    """

    def __init__(
        self,
        save_dir: str | os.PathLike[str],
        process_id: int,
        num_envs: int,
        include_observations: bool = True,
        mode: str = "official_eval",
    ) -> None:
        if mode not in AUDIT_MODES:
            raise ValueError(
                f"Unsupported episode auditor mode {mode!r}; expected {AUDIT_MODES}"
            )
        self.save_dir = Path(save_dir)
        self.process_id = int(process_id)
        self.num_envs = int(num_envs)
        self.include_observations = bool(include_observations)
        self.mode = mode
        self._output_dir = self.save_dir / f"process_{self.process_id:04d}"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._episode_indices = [
            self._next_episode_index(env_idx) for env_idx in range(self.num_envs)
        ]
        self._episodes: list[dict[str, Any] | None] = [None] * self.num_envs

    def start_episode(
        self,
        env_idx: int,
        metadata: dict[str, Any],
        initial_observation: Any = None,
    ) -> None:
        """Start auditing an episode, finalizing an interrupted predecessor."""
        self._check_env_idx(env_idx)
        if self._episodes[env_idx] is not None:
            self._finalize(env_idx, end_reason="external_reset")

        episode_index = self._episode_indices[env_idx]
        episode_id = (
            f"process-{self.process_id:04d}-env-{env_idx:03d}-"
            f"episode-{episode_index:06d}"
        )
        episode: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "timestep_base": 1,
            "episode": {
                "episode_id": episode_id,
                "process_id": self.process_id,
                "env_index": env_idx,
                "episode_index": episode_index,
                "collection_mode": self.mode,
                **_json_value(metadata),
            },
            "trajectory": [],
        }
        if self.include_observations:
            episode["initial_observation"] = _non_visual_observation(
                initial_observation
            )
        self._episodes[env_idx] = episode

    def record_transition(
        self,
        env_idx: int,
        *,
        timestep: int,
        action: Any,
        observation: Any,
        reward: Any,
        raw_task_termination: bool,
        safety_violation_termination: bool,
        severe_collision_termination: bool,
        official_termination: bool,
        emitted_termination: bool,
        truncated: bool,
        predicates: list[dict[str, Any]],
    ) -> Path | None:
        """Append one transition and finalize according to the audit mode."""
        self._check_env_idx(env_idx)
        episode = self._episodes[env_idx]
        if episode is None:
            return None

        transition: dict[str, Any] = {
            "timestep": int(timestep),
            "action": _json_value(action),
            "reward": _json_value(reward),
            "terminations": {
                "raw_task": bool(raw_task_termination),
                "safety_violation": bool(safety_violation_termination),
                "severe_collision": bool(severe_collision_termination),
                "official": bool(official_termination),
                "emitted": bool(emitted_termination),
                "truncated": bool(truncated),
            },
            "predicates": _json_value(predicates),
        }
        if self.include_observations:
            transition["observation"] = _non_visual_observation(observation)
        episode["trajectory"].append(transition)

        if severe_collision_termination:
            return self._finalize(env_idx, end_reason="severe_collision")
        if (
            self.mode == "official_eval"
            and emitted_termination
            and safety_violation_termination
        ):
            return self._finalize(env_idx, end_reason="safety_violation")
        if raw_task_termination:
            return self._finalize(env_idx, end_reason="raw_task_success")
        if truncated:
            return self._finalize(env_idx, end_reason="horizon")
        return None

    def close(self) -> None:
        """Persist any episode still active when the environment closes."""
        for env_idx, episode in enumerate(self._episodes):
            if episode is not None and episode["trajectory"]:
                self._finalize(env_idx, end_reason="environment_close")

    def _finalize(self, env_idx: int, end_reason: str) -> Path | None:
        episode = self._episodes[env_idx]
        if episode is None:
            return None
        trajectory = episode["trajectory"]
        if not trajectory:
            self._episodes[env_idx] = None
            return None

        first_raw_success = next(
            (
                step["timestep"]
                for step in trajectory
                if step["terminations"]["raw_task"]
            ),
            None,
        )
        first_violation = next(
            (
                step["timestep"]
                for step in trajectory
                if step["terminations"]["safety_violation"]
            ),
            None,
        )
        first_severe_collision = next(
            (
                step["timestep"]
                for step in trajectory
                if step["terminations"]["severe_collision"]
            ),
            None,
        )
        raw_success = first_raw_success is not None
        safety_violation = first_violation is not None
        severe_collision = first_severe_collision is not None
        quadrant = {
            (True, False): "Q1",
            (True, True): "Q2",
            (False, False): "Q3",
            (False, True): "Q4",
        }[(raw_success, safety_violation)]
        episode["outcome"] = {
            "success": raw_success,
            "unsafe": safety_violation,
            "raw_task_success": raw_success,
            "safety_violation": safety_violation,
            "severe_collision": severe_collision,
            "safe_task_success": raw_success and not safety_violation,
            "quadrant": quadrant,
            "first_raw_task_success_timestep": first_raw_success,
            "first_violation_timestep": first_violation,
            "first_severe_collision_timestep": first_severe_collision,
            "end_reason": end_reason,
            "num_transitions": len(trajectory),
        }
        first_violation_index = next(
            (
                index
                for index, step in enumerate(trajectory)
                if step["terminations"]["safety_violation"]
            ),
            None,
        )
        episode["success"] = raw_success
        episode["unsafe"] = safety_violation
        episode["trajectory_after_violation"] = (
            trajectory[first_violation_index:]
            if first_violation_index is not None
            else []
        )

        episode_id = episode["episode"]["episode_id"]
        output_path = self._output_dir / f"{episode_id}.json"
        temporary_path = output_path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(episode, file, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary_path, output_path)

        self._episodes[env_idx] = None
        self._episode_indices[env_idx] += 1
        return output_path

    def _next_episode_index(self, env_idx: int) -> int:
        prefix = f"process-{self.process_id:04d}-env-{env_idx:03d}-episode-"
        existing_indices = []
        for path in self._output_dir.glob(f"{prefix}*.json"):
            raw_index = path.stem.removeprefix(prefix)
            if raw_index.isdigit():
                existing_indices.append(int(raw_index))
        return max(existing_indices, default=-1) + 1

    def _check_env_idx(self, env_idx: int) -> None:
        if not 0 <= env_idx < self.num_envs:
            raise IndexError(f"env_idx={env_idx} is outside [0, {self.num_envs})")
