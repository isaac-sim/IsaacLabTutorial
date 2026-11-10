# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manual target mapping and reset contracts, independent of a GPU/viewer."""

import numpy as np
import pytest

from isaaclab_tutorial.locomotion.playground.controls import JointControls


def test_targets_follow_named_order_and_invert_action_mapping():
    controls = JointControls(
        names=("right_ankle", "left_knee"),
        default=np.array([0.4, -0.2]),
        limits=np.array([[-1.0, 1.0], [-0.8, 0.3]]),
        scale=np.array([0.5, 2.0]),
    )
    controls.set_target("left_knee", 0.2)
    controls.set_target("right_ankle", 0.6)
    np.testing.assert_allclose(controls.actions(encoder_bias=np.array([0.05, -0.1])), [0.5, 0.15])


def test_invalid_target_cannot_replace_last_valid_target_and_reset_restores_stance():
    controls = JointControls(("knee",), np.array([0.1]), np.array([[-0.5, 0.8]]), np.array([1.0]))
    controls.set_target("knee", 0.6)
    for invalid in (float("nan"), float("inf"), 0.9):
        with pytest.raises(ValueError):
            controls.set_target("knee", invalid)
    np.testing.assert_allclose(controls.targets, [0.6])
    controls.restore()
    np.testing.assert_allclose(controls.targets, [0.1])
    np.testing.assert_allclose(controls.actions(np.zeros(1)), [0.0])


def test_pause_step_and_reset_requests_are_consumed_once():
    controls = JointControls(("knee",), np.zeros(1), np.array([[-1.0, 1.0]]), np.ones(1))
    controls.paused = True
    assert not controls.consume_step()
    controls.step_requested = True
    assert controls.consume_step()
    assert not controls.consume_step()
    controls.reset_requested = True
    assert controls.consume_reset()
    assert not controls.consume_reset()
    controls.paused = False
    assert controls.consume_step()


def test_commands_keep_training_bounds_and_stand_does_not_write_joint_targets():
    controls = JointControls(
        ("knee",),
        np.zeros(1),
        np.array([[-1.0, 1.0]]),
        np.ones(1),
        command_limits=np.array([[-0.4, 0.4], [-0.3, 0.3], [-1.0, 1.0]]),
    )
    controls.set_target("knee", 0.2)
    controls.set_command(0, 0.3)
    controls.set_command(2, -0.5)
    for axis, value in ((0, 0.5), (1, float("nan")), (3, 0.1)):
        with pytest.raises(ValueError):
            controls.set_command(axis, value)
    np.testing.assert_array_equal(controls.commands, [0.3, 0, -0.5])
    controls.stand()
    np.testing.assert_array_equal(controls.commands, [0, 0, 0])
    np.testing.assert_array_equal(controls.targets, [0.2])
    controls.ended = True
    controls.step_requested = True
    assert not controls.consume_step()
