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

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rlinf.envs.sim.libero.safety_auditor import (
    SCHEMA_VERSION,
    LiberoSafetyEpisodeAuditor,
    audit_predicate_instances,
)


def _predicate(instance_id: str, value: bool) -> dict:
    return {
        "instance_id": instance_id,
        "source": "constraint",
        "predicate": "checkcontact",
        "arguments": ["object_a", "obstacle_a"],
        "semantics": "violation",
        "value": value,
    }


def _start(auditor: LiberoSafetyEpisodeAuditor) -> None:
    auditor.start_episode(
        0,
        metadata={
            "task_suite": "obstacle_avoidance",
            "safety_level": "L0",
            "task_id": 3,
            "trial_id": 7,
            "reset_state_id": 11,
            "instruction": "put the mug in the microwave",
        },
        initial_observation={
            "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
            "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
        },
    )


def test_predicate_audit_preserves_duplicate_instances():
    class FakeTask:
        parsed_problem = {
            "goal_state": [["on", "mug", "plate"]],
            "constraints": [
                ["checkcontact", "mug", "obstacle_a"],
                ["checkcontact", "mug", "obstacle_b"],
            ],
        }

        @staticmethod
        def _eval_predicate(instance):
            return instance[-1] in {"plate", "obstacle_a"}

    predicates = audit_predicate_instances(SimpleNamespace(env=FakeTask()))

    assert [predicate["instance_id"] for predicate in predicates] == [
        "goal:000",
        "constraint:000",
        "constraint:001",
    ]
    assert [predicate["predicate"] for predicate in predicates] == [
        "on",
        "checkcontact",
        "checkcontact",
    ]
    assert [predicate["value"] for predicate in predicates] == [True, True, False]


def test_auditor_continues_after_violation_and_writes_q2(tmp_path: Path):
    auditor = LiberoSafetyEpisodeAuditor(tmp_path, process_id=2, num_envs=1)
    _start(auditor)
    duplicate_predicates = [
        _predicate("constraint:000", True),
        _predicate("constraint:001", False),
    ]

    first_path = auditor.record_transition(
        0,
        timestep=1,
        action=np.zeros(7),
        observation={"robot0_eef_pos": np.array([0.2, 0.2, 0.3])},
        reward=0.0,
        raw_task_termination=False,
        safety_violation_termination=True,
        official_termination=True,
        emitted_termination=False,
        truncated=False,
        predicates=duplicate_predicates,
    )
    second_path = auditor.record_transition(
        0,
        timestep=2,
        action=np.ones(7),
        observation={"robot0_eef_pos": np.array([0.3, 0.2, 0.3])},
        reward=0.0,
        raw_task_termination=False,
        safety_violation_termination=False,
        official_termination=False,
        emitted_termination=False,
        truncated=False,
        predicates=[_predicate("constraint:000", False)],
    )
    output_path = auditor.record_transition(
        0,
        timestep=3,
        action=np.full(7, 2.0),
        observation={"robot0_eef_pos": np.array([0.4, 0.2, 0.3])},
        reward=1.0,
        raw_task_termination=True,
        safety_violation_termination=False,
        official_termination=True,
        emitted_termination=True,
        truncated=False,
        predicates=[_predicate("constraint:000", False)],
    )

    assert first_path is None
    assert second_path is None
    assert output_path is not None
    with output_path.open(encoding="utf-8") as file:
        record = json.load(file)

    assert record["schema_version"] == SCHEMA_VERSION
    assert len(record["trajectory"]) == 3
    assert len(record["trajectory"][0]["predicates"]) == 2
    assert record["trajectory"][0]["terminations"] == {
        "raw_task": False,
        "safety_violation": True,
        "official": True,
        "emitted": False,
        "truncated": False,
    }
    assert record["outcome"] == {
        "raw_task_success": True,
        "safety_violation": True,
        "safe_task_success": False,
        "quadrant": "Q2",
        "first_raw_task_success_timestep": 3,
        "first_violation_timestep": 1,
        "end_reason": "raw_task_success",
        "num_transitions": 3,
    }
    assert "agentview_image" not in record["initial_observation"]
    assert record["initial_observation"]["robot0_eef_pos"] == [0.1, 0.2, 0.3]


def test_auditor_writes_safe_failure_at_horizon(tmp_path: Path):
    auditor = LiberoSafetyEpisodeAuditor(
        tmp_path,
        process_id=0,
        num_envs=1,
        include_observations=False,
    )
    _start(auditor)

    output_path = auditor.record_transition(
        0,
        timestep=1,
        action=[0.0] * 7,
        observation={"agentview_image": np.zeros((2, 2, 3))},
        reward=0.0,
        raw_task_termination=False,
        safety_violation_termination=False,
        official_termination=False,
        emitted_termination=False,
        truncated=True,
        predicates=[],
    )

    assert output_path is not None
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["outcome"]["quadrant"] == "Q3"
    assert record["outcome"]["first_violation_timestep"] is None
    assert "initial_observation" not in record
    assert "observation" not in record["trajectory"][0]

    resumed_auditor = LiberoSafetyEpisodeAuditor(
        tmp_path,
        process_id=0,
        num_envs=1,
        include_observations=False,
    )
    _start(resumed_auditor)
    resumed_path = resumed_auditor.record_transition(
        0,
        timestep=1,
        action=[0.0] * 7,
        observation={},
        reward=0.0,
        raw_task_termination=False,
        safety_violation_termination=False,
        official_termination=False,
        emitted_termination=False,
        truncated=True,
        predicates=[],
    )
    assert resumed_path is not None
    assert resumed_path != output_path
    assert resumed_path.name.endswith("episode-000001.json")
