# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen observation inspection must not change the controller's live inputs."""

import numpy as np
import pytest

from isaaclab_tutorial.locomotion.playground.inspector import ObservationInspector


def test_frozen_display_retains_its_time_and_distinguishes_absolute_sensor_from_policy_offset():
    inspector = ObservationInspector({"policy": [("imu", 6), ("joint_pos", 2)], "critic": [("base_lin_vel", 3)]})
    actor = np.array([0, 0, 0, 0, 0, -1, 0.2, -0.1], dtype=float)
    inspector.update({"policy": actor, "critic": np.array([0.1, 0, 0])}, {"joint_pos": np.array([1.2, -0.4])}, 0.02)
    np.testing.assert_array_equal(inspector.values("policy", "joint_pos"), [0.2, -0.1])
    np.testing.assert_array_equal(inspector.values("policy", "joint_pos", raw=True), [1.2, -0.4])
    inspector.frozen = True
    actor[6] = 0.8
    inspector.update({"policy": actor, "critic": np.ones(3)}, {"joint_pos": np.ones(2)}, 0.04)
    assert inspector.time_s == 0.02
    np.testing.assert_array_equal(inspector.values("policy", "joint_pos"), [0.2, -0.1])
    np.testing.assert_array_equal(actor, [0, 0, 0, 0, 0, -1, 0.8, -0.1])
    inspector.frozen = False
    inspector.update({"policy": actor, "critic": np.ones(3)}, {"joint_pos": np.ones(2)}, 0.04)
    assert inspector.time_s == 0.04
    np.testing.assert_array_equal(inspector.values("policy", "joint_pos"), [0.8, -0.1])


def test_wrong_observation_width_is_not_presented_under_plausible_term_names():
    inspector = ObservationInspector({"policy": [("joint_pos", 2), ("joint_vel", 2)]})
    with pytest.raises(ValueError, match="width"):
        inspector.update({"policy": np.zeros(3)}, {}, 0.0)
