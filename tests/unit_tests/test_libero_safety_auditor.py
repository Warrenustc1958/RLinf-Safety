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
    detect_severe_collisions,
    select_emitted_terminations,
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


def test_shadow_mode_only_emits_raw_success_or_severe_collision():
    raw_task = np.array([False, True, False])
    official = np.array([True, True, True])
    severe = np.array([False, False, True])

    assert select_emitted_terminations(
        "official_eval", raw_task, official, severe
    ).tolist() == [True, True, True]
    assert select_emitted_terminations(
        "shadow_collect", raw_task, official, severe
    ).tolist() == [False, True, True]


def test_severe_collision_uses_predicate_and_persistence_guards():
    predicate_audits = [
        [
            {
                "source": "constraint",
                "predicate": "checkgripperforce",
                "value": True,
            }
        ],
        [_predicate("constraint:000", True)],
    ]
    severe, streak = detect_severe_collisions(
        predicate_audits=predicate_audits,
        constraint_violations=np.array([True, True]),
        consecutive_violation_steps=np.array([0, 7]),
        hard_stop_predicates={"checkgripperforce"},
        max_consecutive_violation_steps=8,
    )

    assert severe.tolist() == [True, True]
    assert streak.tolist() == [1, 8]


def test_auditor_continues_after_violation_and_writes_q2(tmp_path: Path):
    auditor = LiberoSafetyEpisodeAuditor(
        tmp_path,
        process_id=2,
        num_envs=1,
        mode="shadow_collect",
    )
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
        safety_violation_termination=False,
        severe_collision_termination=False,
        official_termination=False,
        emitted_termination=False,
        truncated=False,
        predicates=[_predicate("constraint:000", False)],
    )
    second_path = auditor.record_transition(
        0,
        timestep=2,
        action=np.ones(7),
        observation={"robot0_eef_pos": np.array([0.3, 0.2, 0.3])},
        reward=0.0,
        raw_task_termination=False,
        safety_violation_termination=True,
        severe_collision_termination=False,
        official_termination=True,
        emitted_termination=False,
        truncated=False,
        predicates=duplicate_predicates,
    )
    output_path = auditor.record_transition(
        0,
        timestep=3,
        action=np.full(7, 2.0),
        observation={"robot0_eef_pos": np.array([0.4, 0.2, 0.3])},
        reward=1.0,
        raw_task_termination=True,
        safety_violation_termination=False,
        severe_collision_termination=False,
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
    assert len(record["trajectory"][1]["predicates"]) == 2
    assert record["trajectory"][1]["terminations"] == {
        "raw_task": False,
        "safety_violation": True,
        "severe_collision": False,
        "official": True,
        "emitted": False,
        "truncated": False,
    }
    assert record["outcome"] == {
        "success": True,
        "unsafe": True,
        "raw_task_success": True,
        "safety_violation": True,
        "severe_collision": False,
        "safe_task_success": False,
        "quadrant": "Q2",
        "first_raw_task_success_timestep": 3,
        "first_violation_timestep": 2,
        "first_severe_collision_timestep": None,
        "end_reason": "raw_task_success",
        "num_transitions": 3,
    }
    assert record["success"] is True
    assert record["unsafe"] is True
    assert [step["timestep"] for step in record["trajectory_after_violation"]] == [
        2,
        3,
    ]
    assert "agentview_image" not in record["initial_observation"]
    assert record["initial_observation"]["robot0_eef_pos"] == [0.1, 0.2, 0.3]


def test_official_violation_and_shadow_severe_collision_hard_stop(tmp_path: Path):
    official = LiberoSafetyEpisodeAuditor(
        tmp_path / "official", process_id=0, num_envs=1
    )
    _start(official)
    official_path = official.record_transition(
        0,
        timestep=1,
        action=[0.0] * 7,
        observation={},
        reward=0.0,
        raw_task_termination=False,
        safety_violation_termination=True,
        severe_collision_termination=False,
        official_termination=True,
        emitted_termination=True,
        truncated=False,
        predicates=[_predicate("constraint:000", True)],
    )
    assert official_path is not None
    official_record = json.loads(official_path.read_text(encoding="utf-8"))
    assert official_record["outcome"]["end_reason"] == "safety_violation"

    shadow = LiberoSafetyEpisodeAuditor(
        tmp_path / "shadow",
        process_id=0,
        num_envs=1,
        mode="shadow_collect",
    )
    _start(shadow)
    severe_path = shadow.record_transition(
        0,
        timestep=1,
        action=[0.0] * 7,
        observation={},
        reward=0.0,
        raw_task_termination=False,
        safety_violation_termination=True,
        severe_collision_termination=True,
        official_termination=True,
        emitted_termination=True,
        truncated=False,
        predicates=[_predicate("constraint:000", True)],
    )
    assert severe_path is not None
    severe_record = json.loads(severe_path.read_text(encoding="utf-8"))
    assert severe_record["outcome"]["end_reason"] == "severe_collision"
    assert severe_record["outcome"]["severe_collision"] is True


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
        severe_collision_termination=False,
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
        severe_collision_termination=False,
        official_termination=False,
        emitted_termination=False,
        truncated=True,
        predicates=[],
    )
    assert resumed_path is not None
    assert resumed_path != output_path
    assert resumed_path.name.endswith("episode-000001.json")
