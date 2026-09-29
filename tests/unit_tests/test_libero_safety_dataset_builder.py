# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rlinf.data.datasets.world_model import NpyTrajectoryDatasetWrapper
from rlinf.envs.sim.libero.safety_auditor import LiberoSafetyEpisodeAuditor
from toolkits.libero_safety.build_dataset import build_dataset


def _observation(value: int) -> dict:
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3) + value
    return {
        "agentview_image": image,
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([-1.0, -1.0], dtype=np.float32),
    }


def _predicate(value: bool) -> dict:
    return {
        "instance_id": "constraint:000",
        "source": "constraint",
        "predicate": "checkcontact",
        "arguments": ["mug", "obstacle"],
        "semantics": "violation",
        "value": value,
    }


def _metadata(**extra) -> dict:
    return {
        "task_suite": "obstacle_avoidance",
        "safety_level": "L0",
        "task_id": 1,
        "trial_id": 2,
        "reset_state_id": 3,
        "instruction": "put the mug in the basket",
        **extra,
    }


def _record(
    auditor: LiberoSafetyEpisodeAuditor,
    timestep: int,
    *,
    violation: bool,
    success: bool,
):
    return auditor.record_transition(
        0,
        timestep=timestep,
        action=np.full(7, timestep, dtype=np.float32),
        observation=_observation(timestep),
        reward=float(success),
        raw_task_termination=success,
        safety_violation_termination=violation,
        severe_collision_termination=False,
        official_termination=violation or success,
        emitted_termination=success,
        truncated=False,
        predicates=[_predicate(violation)],
    )


def test_builder_exports_aligned_wan_and_sam_data(tmp_path: Path):
    audit_root = tmp_path / "audits"
    auditor = LiberoSafetyEpisodeAuditor(
        audit_root,
        process_id=0,
        num_envs=1,
        mode="shadow_collect",
        save_visual_observations=True,
    )

    auditor.start_episode(0, _metadata(), _observation(0))
    q1_path = _record(auditor, 1, violation=False, success=True)
    assert q1_path is not None
    q1_id = json.loads(q1_path.read_text())["episode"]["episode_id"]

    auditor.start_episode(
        0,
        _metadata(
            episode_kind="counterfactual_branch",
            root_episode_id=q1_id,
            branch_id=f"{q1_id}-branch-000",
            alpha=0.5,
            perturbation_range=[0.2, 0.8],
        ),
        _observation(10),
    )
    assert _record(auditor, 1, violation=True, success=False) is None
    q2_path = _record(auditor, 2, violation=False, success=True)
    assert q2_path is not None

    q1_record = json.loads(q1_path.read_text())
    visual_path = q1_path.parent / q1_record["visual_trajectory"]["path"]
    with np.load(visual_path, allow_pickle=False) as visual:
        assert visual["agentview_image"].shape == (2, 2, 3, 3)
        np.testing.assert_array_equal(
            visual["agentview_image"][0], _observation(0)["agentview_image"][::-1, ::-1]
        )

    output = tmp_path / "dataset"
    summary = build_dataset(
        audit_roots=[audit_root],
        output_dir=output,
        quadrants=("Q1", "Q2"),
        train_fraction=1.0,
        val_fraction=0.0,
        min_per_quadrant={"Q1": 1, "Q2": 1},
    )

    assert summary["quadrant_counts"] == {"Q1": 1, "Q2": 1, "Q3": 0, "Q4": 0}
    assert summary["ready_for_wan"] is True
    manifest = [
        json.loads(line)
        for line in (output / "manifest.jsonl").read_text().splitlines()
    ]
    assert len(manifest) == 2
    assert {item["quadrant"] for item in manifest} == {"Q1", "Q2"}

    q2_manifest = next(item for item in manifest if item["quadrant"] == "Q2")
    trajectory = np.load(output / q2_manifest["wan_path"], allow_pickle=True)
    assert len(trajectory) == 2
    assert trajectory[0]["delta_action"].shape == (7,)
    assert trajectory[0]["init_ee_pose"].shape == (8,)
    assert trajectory[0]["post_action_violation"] is True

    # The output is directly consumable by RLinf's existing Wan reset dataset.
    dataset = NpyTrajectoryDatasetWrapper(str(output / "wan" / "train"))
    item = dataset[0]
    assert item["start_items"][0]["image"].shape == (3, 2, 3)
    assert item["start_items"][0]["action"].shape == (7,)

    sam_labels = [
        json.loads(line)
        for line in (output / "sam" / "train.jsonl").read_text().splitlines()
    ]
    q2_labels = [item for item in sam_labels if item["quadrant"] == "Q2"]
    assert q2_labels[0]["post_action_violation"] is True
    assert q2_labels[0]["active_constraint_instance_ids"] == ["constraint:000"]
    assert q2_labels[0]["time_to_next_violation_steps"] == 0
    assert q2_labels[1]["unsafe_so_far"] is True


def test_builder_refuses_missing_visual_sidecar_and_quadrant_shortfall(
    tmp_path: Path,
):
    audit_root = tmp_path / "audits"
    auditor = LiberoSafetyEpisodeAuditor(
        audit_root,
        process_id=0,
        num_envs=1,
        mode="shadow_collect",
    )
    auditor.start_episode(0, _metadata(), _observation(0))
    assert _record(auditor, 1, violation=False, success=True) is not None

    with pytest.raises(ValueError, match="no visual_trajectory"):
        build_dataset(
            audit_roots=[audit_root],
            output_dir=tmp_path / "dataset",
            quadrants=("Q1",),
            dry_run=True,
        )

    visual_root = tmp_path / "multi_run_audits" / "seed_0"
    visual_auditor = LiberoSafetyEpisodeAuditor(
        visual_root,
        process_id=0,
        num_envs=1,
        mode="shadow_collect",
        save_visual_observations=True,
    )
    visual_auditor.start_episode(0, _metadata(), _observation(0))
    assert _record(visual_auditor, 1, violation=False, success=True) is not None
    with pytest.raises(ValueError, match="shortfalls"):
        build_dataset(
            audit_roots=[visual_root],
            output_dir=tmp_path / "dataset",
            quadrants=("Q1", "Q2"),
            min_per_quadrant={"Q2": 1},
            dry_run=True,
        )

    # Auditor episode IDs restart in each rollout directory. The builder must
    # namespace them so independent seeds can be accumulated safely.
    second_visual_root = tmp_path / "multi_run_audits" / "seed_1"
    second_auditor = LiberoSafetyEpisodeAuditor(
        second_visual_root,
        process_id=0,
        num_envs=1,
        mode="shadow_collect",
        save_visual_observations=True,
    )
    second_auditor.start_episode(0, _metadata(), _observation(0))
    assert _record(second_auditor, 1, violation=False, success=True) is not None
    summary = build_dataset(
        audit_roots=[tmp_path / "multi_run_audits"],
        output_dir=tmp_path / "dataset",
        quadrants=("Q1",),
        dry_run=True,
    )
    assert summary["episodes_selected"] == 2
