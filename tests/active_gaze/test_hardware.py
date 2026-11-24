"""Hardware unit conversions, travel checks, and torque shutdown without a robot."""

from types import SimpleNamespace

import numpy as np
import pytest

from isaaclab_tutorial.active_gaze.deployment.control import JOINTS
from isaaclab_tutorial.active_gaze.deployment.hardware import JointMap, require_start_pose

pytestmark = pytest.mark.kitless


def mapping():
    return JointMap(
        {
            "units": "lerobot_degrees_and_gripper_percent",
            "verified": False,
            "joints": {name: {"scale": -0.02, "offset": 0.1, "limits": [-1, 1]} for name in JOINTS},
        }
    )


def test_joint_mapping_roundtrip_negative_direction_and_limits():
    m = mapping()
    native = dict.fromkeys(JOINTS, 4.0)
    commands = m.commands(m.positions(native))
    np.testing.assert_allclose(list(commands.values()), [4] * 6)
    clipped = m.commands(np.full(6, 2.0))
    np.testing.assert_allclose(m.positions({name: clipped[f"{name}.pos"] for name in JOINTS}), [1] * 6)
    with pytest.raises(ValueError, match="finite"):
        m.commands(np.full(6, np.nan))


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

    from isaaclab_tutorial.active_gaze.deployment.hardware import finish_motor_control

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

    from isaaclab_tutorial.active_gaze.deployment.hardware import finish_motor_control

    robot = SimpleNamespace(bus=Mock())
    getattr(robot.bus, operation).side_effect = RuntimeError("Bus failed")
    with pytest.raises(RuntimeError, match="Bus failed"):
        finish_motor_control(robot, hold=True)
    robot.bus.disconnect.assert_called_once_with(disable_torque=True)
