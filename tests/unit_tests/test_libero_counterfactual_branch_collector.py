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

import numpy as np
import pytest

from rlinf.envs.sim.libero.counterfactual_branch_collector import (
    CounterfactualBranchCollector,
)


def _collector(**overrides):
    kwargs = {
        "num_envs": 1,
        "seed": 7,
        "process_id": 2,
        "risk_window_steps": 2,
        "max_branches_per_root": 6,
        "perturbation_range": (0.25, 0.75),
        "action_dimensions": (0, 1, 2, 3, 4, 5),
    }
    kwargs.update(overrides)
    return CounterfactualBranchCollector(**kwargs)


def test_branch_search_samples_q1_tail_and_preserves_gripper():
    collector = _collector()
    for timestep in range(1, 4):
        collector.record_root_step(
            0,
            timestep=timestep,
            simulator_state={"qpos": np.array([timestep], dtype=np.float64)},
            action=np.array([0.1] * 6 + [-1.0]),
        )

    branch = collector.begin_search(0, root_episode_id="root-007")

    assert branch is not None
    assert branch.root_episode_id == "root-007"
    assert branch.branch_id == "root-007-branch-000"
    assert branch.root_timestep in {2, 3}
    assert 0.25 <= branch.alpha <= 0.75
    assert branch.perturbation_range == (0.25, 0.75)
    assert branch.perturbed_action[-1] == -1.0
    assert np.all(branch.perturbed_action >= -1.0)
    assert np.all(branch.perturbed_action <= 1.0)
    np.testing.assert_allclose(
        collector.consume_perturbed_action(0), branch.perturbed_action
    )
    assert collector.consume_perturbed_action(0) is None
    assert branch.metadata()["root_episode_id"] == "root-007"


def test_branch_search_retries_until_q2_and_q4_are_found():
    collector = _collector()
    collector.record_root_step(
        0,
        timestep=5,
        simulator_state={"qpos": np.array([1.0])},
        action=np.zeros(7),
    )
    first = collector.begin_search(0, root_episode_id="root")
    assert first is not None

    second = collector.complete_branch(0, quadrant="Q1")
    assert second is not None
    assert second.branch_id.endswith("branch-001")

    third = collector.complete_branch(0, quadrant="Q2")
    assert third is not None
    assert third.branch_id.endswith("branch-002")

    assert collector.complete_branch(0, quadrant="Q4") is None
    assert not collector.is_running_branch(0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"risk_window_steps": 0}, "risk_window_steps"),
        ({"max_branches_per_root": 0}, "max_branches_per_root"),
        ({"perturbation_range": (-0.1, 0.5)}, "perturbation_range"),
        ({"target_quadrants": frozenset({"Q1"})}, "target_quadrants"),
    ],
)
def test_branch_collector_rejects_invalid_configuration(overrides, message):
    with pytest.raises(ValueError, match=message):
        _collector(**overrides)
