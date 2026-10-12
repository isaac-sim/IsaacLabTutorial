"""Offline checks of real-robot preprocessing and feedback command semantics."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("cv2")

from isaaclab_tutorial.utils.deploy import JOINTS, JointMap, VisualPolicy, require_start_pose  # noqa: E402


def mapping():
    return JointMap(
        {
            "units": "lerobot_degrees_and_gripper_percent",
            "verified": False,
            "joints": {name: {"scale": -0.02, "offset": 0.1, "limits": [-1, 1]} for name in JOINTS},
        }
    )


def policy(shoulder_scale=0.033):
    runtime = SimpleNamespace()

    def infer(inputs):
        runtime.inputs = inputs
        return {"policy/action": torch.tensor([[2.0, -2.0, 0.5, 0.0, 0.0, 0.0]])}

    runtime.run_policy = infer
    return VisualPolicy(
        runtime,
        {
            "joint_order": list(JOINTS),
            "joint_units": "radians",
            "policy_hz": 30,
            "relative_target_hz": 120,
            "proprioception": ["joint_pos", "joint_vel", "joint_target", "previous_clipped_action"],
            "history_order": "oldest_first_repeat_initial_frame",
            "history_length": 2,
            "image_shape": [1, 6, 48, 64],
            "image_preprocessing": "RGB_uint8_divided_by_255",
            "action_scale": [0.033, shoulder_scale, 0.033, 0.033, 0.033, 0.02],
        },
        mapping(),
    )


@pytest.mark.parametrize("shoulder_scale", [0.033, 0.04])
def test_image_history_proprioception_and_feedback_targets(shoulder_scale):
    p = policy(shoulder_scale)
    q = np.zeros(6)
    p.infer(np.full((480, 640, 3), 255, np.uint8), q, q, q)
    torch.testing.assert_close(p.runtime.inputs["policy/wrist_rgb"], torch.ones(1, 6, 48, 64))
    np.testing.assert_allclose(p.target(q), [0.033, -shoulder_scale, 0.0165, 0, 0, 0])
    np.testing.assert_allclose(p.target(q + 0.1), [0.133, 0.1 - shoulder_scale, 0.1165, 0.1, 0.1, 0.1])
    p.infer(np.zeros((480, 640, 3), np.uint8), q, q + 20, q + 0.2)
    image = p.runtime.inputs["policy/wrist_rgb"]
    assert image[:, :3].eq(1).all() and image[:, 3:].eq(0).all()
    obs = p.runtime.inputs["policy/proprioception"]
    torch.testing.assert_close(obs[0, 6:12], torch.full((6,), 12.0))
    torch.testing.assert_close(obs[0, 12:18], torch.full((6,), 0.2))
    torch.testing.assert_close(obs[0, 18:], torch.tensor([1.0, -1.0, 0.5, 0.0, 0.0, 0.0]))


def test_joint_mapping_roundtrip_negative_direction_and_limits():
    m = mapping()
    native = dict.fromkeys(JOINTS, 4.0)
    commands = m.commands(m.positions(native))
    np.testing.assert_allclose(list(commands.values()), [4] * 6)
    clipped = m.commands(np.full(6, 2.0))
    np.testing.assert_allclose(m.positions({name: clipped[f"{name}.pos"] for name in JOINTS}), [1] * 6)
    with pytest.raises(ValueError, match="finite"):
        m.commands(np.full(6, np.nan))


def test_rejects_wrong_camera_shape_and_nonfinite_actor_output():
    p = policy()
    q = np.zeros(6)
    with pytest.raises(ValueError, match="aspect ratio"):
        p.infer(np.zeros((480, 1280, 3), np.uint8), q, q, q)
    p.runtime.run_policy = lambda inputs: {"policy/action": torch.full((1, 6), float("nan"))}
    with pytest.raises(ValueError, match="invalid actions"):
        p.infer(np.zeros((48, 64, 3), np.uint8), q, q, q)


def test_execution_rejects_pose_outside_common_travel_before_clipping():
    m = mapping()
    m.require_in_limits(np.zeros(6))
    measured = np.zeros(6)
    measured[2] = 1.2
    with pytest.raises(ValueError, match="elbow_flex"):
        m.require_in_limits(measured)
    with pytest.raises(ValueError, match="Measured pose"):
        m.require_in_limits(np.full(6, np.nan))


def test_start_pose_rejects_wrong_home_and_invalid_reference():
    reference = {"position": [0.1] * 6, "tolerance_rad": 0.035}
    require_start_pose(np.full(6, 0.12), reference)
    with pytest.raises(ValueError, match="home pose"):
        require_start_pose(np.zeros(6), reference)
    with pytest.raises(ValueError, match="home pose"):
        require_start_pose(np.full(6, np.nan), reference)
    with pytest.raises(ValueError, match="reference"):
        require_start_pose(np.zeros(6), {"position": [0] * 5, "tolerance_rad": 0.1})


@pytest.mark.parametrize("hold", [False, True])
def test_diagnostic_stop_holds_measured_raw_position_only_when_requested(hold):
    from unittest.mock import Mock

    from isaaclab_tutorial.utils.deploy import finish_motor_control

    robot = SimpleNamespace(bus=Mock())
    position = {name: 2048 for name in JOINTS}
    robot.bus.sync_read.return_value = position
    finish_motor_control(robot, hold=hold)
    if hold:
        robot.bus.sync_read.assert_called_once_with("Present_Position", normalize=False)
        robot.bus.sync_write.assert_called_once_with("Goal_Position", position, normalize=False)
    else:
        robot.bus.sync_read.assert_not_called()
        robot.bus.sync_write.assert_not_called()
    robot.bus.disconnect.assert_called_once_with(disable_torque=not hold)


@pytest.mark.parametrize("operation", ["sync_read", "sync_write"])
def test_diagnostic_hold_failure_releases_torque(operation):
    from unittest.mock import Mock

    from isaaclab_tutorial.utils.deploy import finish_motor_control

    robot = SimpleNamespace(bus=Mock())
    getattr(robot.bus, operation).side_effect = RuntimeError("Bus failed")
    with pytest.raises(RuntimeError, match="Bus failed"):
        finish_motor_control(robot, hold=True)
    robot.bus.disconnect.assert_called_once_with(disable_torque=True)


def test_target_limiter_bounds_stall_reversal_and_delayed_tick():
    from isaaclab_tutorial.utils.deploy import TargetLimiter

    limiter = TargetLimiter(np.zeros(6), 5, 2)
    measured = np.zeros(6)
    previous = measured.copy()
    for _ in range(1000):
        target = limiter.limit(np.ones(6), measured, 1 / 120)
        assert np.max(np.abs(target - previous)) <= np.radians(5 / 120) + 1e-12
        assert np.max(np.abs(target - measured)) <= np.radians(2) + 1e-12
        previous = target
    np.testing.assert_allclose(target, np.full(6, np.radians(2)))
    reversed_target = limiter.limit(-np.ones(6), measured, 1.0)
    np.testing.assert_allclose(target - reversed_target, np.full(6, np.radians(5 / 120)))
    with pytest.raises(RuntimeError, match="envelope"):
        limiter.limit(np.ones(6), np.ones(6), 1 / 120)


@pytest.mark.parametrize("rate,offset", [(0, 2), (5, -1), (float("nan"), 2), (5, float("inf"))])
def test_target_limiter_rejects_invalid_limits(rate, offset):
    from isaaclab_tutorial.utils.deploy import TargetLimiter

    with pytest.raises(ValueError):
        TargetLimiter(np.zeros(6), rate, offset)
