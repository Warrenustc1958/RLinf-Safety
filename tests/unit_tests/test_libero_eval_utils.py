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

"""Unit tests for LIBERO evaluation reset-state allocation."""

import numpy as np

from rlinf.envs.sim.libero.utils import (
    build_interleaved_eval_reset_state_ids,
    distribute_reset_state_ids_round_robin,
)


def test_filtered_eval_states_are_interleaved_across_tasks():
    trial_id_bins = [4, 4, 4, 4, 4, 4]
    cumsum_trial_id_bins = np.cumsum(trial_id_bins)

    reset_state_ids = build_interleaved_eval_reset_state_ids(
        trial_id_bins,
        cumsum_trial_id_bins,
        task_ids=[0, 2, 5],
    )

    np.testing.assert_array_equal(
        reset_state_ids,
        [0, 8, 20, 1, 9, 21, 2, 10, 22, 3, 11, 23],
    )


def test_first_fifty_distributed_states_cover_ten_trials_of_five_tasks():
    trial_id_bins = [50] * 15
    cumsum_trial_id_bins = np.cumsum(trial_id_bins)
    reset_state_ids = build_interleaved_eval_reset_state_ids(
        trial_id_bins,
        cumsum_trial_id_bins,
        task_ids=[0, 3, 6, 9, 12],
    )
    distributed = distribute_reset_state_ids_round_robin(reset_state_ids, 5)

    # Five ranks with ten environments each consume the first ten entries of
    # their own row. Reconstruct global round-robin order to inspect coverage.
    evaluated = distributed[:, :10].T.reshape(-1)
    task_ids = np.searchsorted(cumsum_trial_id_bins, evaluated, side="right")
    trial_ids = evaluated - np.concatenate(([0], cumsum_trial_id_bins[:-1]))[task_ids]

    assert np.bincount(task_ids, minlength=15)[[0, 3, 6, 9, 12]].tolist() == [
        10,
        10,
        10,
        10,
        10,
    ]
    assert sorted(set(trial_ids.tolist())) == list(range(10))
