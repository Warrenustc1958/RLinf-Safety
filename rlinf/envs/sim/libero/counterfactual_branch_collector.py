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

"""Counterfactual branch search for safe LIBERO-Safety episodes."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

TARGET_QUADRANTS = frozenset({"Q2", "Q4"})


@dataclass(frozen=True)
class RiskSnapshot:
    """Simulator state and policy action immediately before one root step."""

    timestep: int
    simulator_state: Any
    action: np.ndarray


@dataclass(frozen=True)
class CounterfactualBranch:
    """One reproducible counterfactual branch specification."""

    root_episode_id: str
    branch_id: str
    root_timestep: int
    simulator_state: Any
    alpha: float
    perturbation_range: tuple[float, float]
    original_action: np.ndarray
    perturbed_action: np.ndarray
    perturbation: np.ndarray

    def metadata(self) -> dict[str, Any]:
        """Return JSON-compatible branch metadata for the episode auditor."""
        return {
            "root_episode_id": self.root_episode_id,
            "branch_id": self.branch_id,
            "branch_timestep": self.root_timestep,
            "alpha": self.alpha,
            "perturbation_range": list(self.perturbation_range),
            "original_action": self.original_action.tolist(),
            "perturbed_action": self.perturbed_action.tolist(),
            "perturbation": self.perturbation.tolist(),
        }


class CounterfactualBranchCollector:
    """Search Q2 and Q4 branches from the tail of a Q1 root trajectory.

    The collector owns only branch selection and perturbation state. The caller
    remains responsible for saving/restoring the simulator, executing actions,
    rolling out the policy, and writing episode audit files.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        seed: int,
        process_id: int,
        risk_window_steps: int = 32,
        max_branches_per_root: int = 16,
        perturbation_range: tuple[float, float] = (0.15, 1.0),
        action_dimensions: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
        target_quadrants: frozenset[str] = TARGET_QUADRANTS,
    ) -> None:
        alpha_min, alpha_max = map(float, perturbation_range)
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        if risk_window_steps < 1:
            raise ValueError("risk_window_steps must be at least 1")
        if max_branches_per_root < 1:
            raise ValueError("max_branches_per_root must be at least 1")
        if not 0.0 <= alpha_min <= alpha_max <= 1.0:
            raise ValueError("perturbation_range must satisfy 0 <= min <= max <= 1")
        if not action_dimensions:
            raise ValueError("action_dimensions must not be empty")
        if not target_quadrants or not target_quadrants <= TARGET_QUADRANTS:
            raise ValueError("target_quadrants must be a non-empty subset of Q2/Q4")

        self.num_envs = int(num_envs)
        self.risk_window_steps = int(risk_window_steps)
        self.max_branches_per_root = int(max_branches_per_root)
        self.perturbation_range = (alpha_min, alpha_max)
        self.action_dimensions = tuple(int(index) for index in action_dimensions)
        self.target_quadrants = frozenset(target_quadrants)
        seed_sequence = np.random.SeedSequence([int(seed), int(process_id)])
        self._rngs = [
            np.random.default_rng(child) for child in seed_sequence.spawn(self.num_envs)
        ]
        self._histories = [
            deque(maxlen=self.risk_window_steps) for _ in range(self.num_envs)
        ]
        self._phases = ["root"] * self.num_envs
        self._root_episode_ids: list[str | None] = [None] * self.num_envs
        self._branch_counts = np.zeros(self.num_envs, dtype=np.int32)
        self._found_quadrants = [set() for _ in range(self.num_envs)]
        self._active_branches: list[CounterfactualBranch | None] = [
            None
        ] * self.num_envs
        self._pending_perturbed_actions: list[np.ndarray | None] = [
            None
        ] * self.num_envs

    def reset_root(self, env_idx: int) -> None:
        """Start collecting snapshots for a new candidate root episode."""
        self._check_env_idx(env_idx)
        self._histories[env_idx].clear()
        self._phases[env_idx] = "root"
        self._root_episode_ids[env_idx] = None
        self._branch_counts[env_idx] = 0
        self._found_quadrants[env_idx].clear()
        self._active_branches[env_idx] = None
        self._pending_perturbed_actions[env_idx] = None

    def is_collecting_root(self, env_idx: int) -> bool:
        """Return whether the environment is recording a root trajectory."""
        self._check_env_idx(env_idx)
        return self._phases[env_idx] == "root"

    def is_running_branch(self, env_idx: int) -> bool:
        """Return whether the environment is currently rolling out a branch."""
        self._check_env_idx(env_idx)
        return self._phases[env_idx] == "branch"

    def record_root_step(
        self,
        env_idx: int,
        *,
        timestep: int,
        simulator_state: Any,
        action: np.ndarray,
    ) -> None:
        """Record a bounded pre-action snapshot while the root is active."""
        self._check_env_idx(env_idx)
        if not self.is_collecting_root(env_idx):
            return
        self._histories[env_idx].append(
            RiskSnapshot(
                timestep=int(timestep),
                simulator_state=copy.deepcopy(simulator_state),
                action=np.asarray(action, dtype=np.float64).copy(),
            )
        )

    def begin_search(
        self, env_idx: int, *, root_episode_id: str
    ) -> CounterfactualBranch | None:
        """Begin branching after the recorded root is confirmed as Q1."""
        self._check_env_idx(env_idx)
        if not self._histories[env_idx]:
            self._phases[env_idx] = "finished"
            return None
        self._root_episode_ids[env_idx] = str(root_episode_id)
        self._phases[env_idx] = "branch"
        self._branch_counts[env_idx] = 0
        self._found_quadrants[env_idx].clear()
        return self._sample_branch(env_idx)

    def consume_perturbed_action(self, env_idx: int) -> np.ndarray | None:
        """Consume the one-shot action that starts the active branch."""
        self._check_env_idx(env_idx)
        action = self._pending_perturbed_actions[env_idx]
        self._pending_perturbed_actions[env_idx] = None
        return None if action is None else action.copy()

    def complete_branch(
        self, env_idx: int, *, quadrant: str
    ) -> CounterfactualBranch | None:
        """Record an outcome and return the next branch, if search continues."""
        self._check_env_idx(env_idx)
        if not self.is_running_branch(env_idx):
            raise RuntimeError(f"env {env_idx} has no active counterfactual search")
        if quadrant not in {"Q1", "Q2", "Q3", "Q4"}:
            raise ValueError(f"invalid branch quadrant {quadrant!r}")
        if quadrant in self.target_quadrants:
            self._found_quadrants[env_idx].add(quadrant)
        self._active_branches[env_idx] = None
        self._pending_perturbed_actions[env_idx] = None
        if (
            self._found_quadrants[env_idx] >= self.target_quadrants
            or self._branch_counts[env_idx] >= self.max_branches_per_root
        ):
            self._phases[env_idx] = "finished"
            return None
        return self._sample_branch(env_idx)

    def active_branch(self, env_idx: int) -> CounterfactualBranch | None:
        """Return the active branch specification without consuming it."""
        self._check_env_idx(env_idx)
        return self._active_branches[env_idx]

    def _sample_branch(self, env_idx: int) -> CounterfactualBranch:
        rng = self._rngs[env_idx]
        history = self._histories[env_idx]
        snapshot = history[int(rng.integers(0, len(history)))]
        alpha = float(rng.uniform(*self.perturbation_range))
        original = snapshot.action.copy()
        random_action = rng.uniform(-1.0, 1.0, size=original.shape)
        perturbed = original.copy()
        for action_index in self.action_dimensions:
            if not 0 <= action_index < original.size:
                raise ValueError(
                    f"action dimension {action_index} outside action size {original.size}"
                )
            perturbed[action_index] = (1.0 - alpha) * original[
                action_index
            ] + alpha * random_action[action_index]
        perturbed = np.clip(perturbed, -1.0, 1.0)
        branch_number = int(self._branch_counts[env_idx])
        self._branch_counts[env_idx] += 1
        root_episode_id = self._root_episode_ids[env_idx]
        if root_episode_id is None:
            raise RuntimeError("root episode id is missing")
        branch = CounterfactualBranch(
            root_episode_id=root_episode_id,
            branch_id=f"{root_episode_id}-branch-{branch_number:03d}",
            root_timestep=snapshot.timestep,
            simulator_state=copy.deepcopy(snapshot.simulator_state),
            alpha=alpha,
            perturbation_range=self.perturbation_range,
            original_action=original,
            perturbed_action=perturbed,
            perturbation=perturbed - original,
        )
        self._active_branches[env_idx] = branch
        self._pending_perturbed_actions[env_idx] = perturbed.copy()
        return branch

    def _check_env_idx(self, env_idx: int) -> None:
        if not 0 <= env_idx < self.num_envs:
            raise IndexError(f"env_idx={env_idx} is outside [0, {self.num_envs})")
