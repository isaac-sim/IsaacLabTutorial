"""Homing rate limits, preview behavior and torque ownership using a simulated bus."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from isaaclab_tutorial.utils import home_so101 as h


def setup_robot(monkeypatch, follow=True):
    clock = [0.0]
    monkeypatch.setattr(
        h,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds)),
    )
    writes = []
    native = dict.fromkeys(h.JOINTS, 0.0)

    def read(name, normalize=True):
        return native.copy() if normalize else dict.fromkeys(h.JOINTS, 2048)

    def write(name, values, normalize=True):
        writes.append(("goal", normalize))
        if follow and normalize:
            native.update(values)

    bus = SimpleNamespace(sync_read=read, sync_write=write, disable_torque=lambda: writes.append(("disable",)))
    robot = SimpleNamespace(
        bus=bus,
        configure=lambda: writes.append(("configure",)),
        calibration={n: SimpleNamespace(range_min=0, range_max=4095) for n in h.JOINTS},
    )
    mapping = h.JointMap(
        {
            "units": "lerobot_degrees_and_gripper_percent",
            "verified": False,
            "joints": {n: {"scale": 0.01, "offset": 0.0, "limits": [-1, 1]} for n in h.JOINTS},
        }
    )
    return robot, mapping, writes


def test_homing_step_rate_limit_and_tracking_error():
    q = np.zeros(6)
    goal = np.array([1, -0.5, 0.1, 0, 0, 0.2])
    result = h.homing_step(q, q, goal, 0.01, 0.1)
    np.testing.assert_allclose(result, goal * 0.01)
    np.testing.assert_allclose(h.homing_step(goal - 0.001, goal, goal, 0.01, 0.1), goal)
    with pytest.raises(RuntimeError, match="tracking error"):
        h.homing_step(q, q + 0.2, goal, 0.01, 0.1)
    with pytest.raises(ValueError, match="finite"):
        h.homing_step(q, q * np.nan, goal, 0.01, 0.1)


def test_preview_never_writes_and_unreachable_goal_rejected(monkeypatch):
    robot, mapping, writes = setup_robot(monkeypatch)
    h.home(robot, mapping, np.full(6, 0.1))
    assert writes == []
    with pytest.raises(ValueError, match="calibrated motor travel"):
        h.home(robot, mapping, np.full(6, -0.1), execute=True)  # Gripper cannot go below native zero.
    assert writes == []


def test_success_seeds_goal_before_configuration_and_keeps_torque(monkeypatch):
    robot, mapping, writes = setup_robot(monkeypatch)
    h.home(robot, mapping, np.full(6, 0.1), execute=True)
    assert writes[:3] == [("disable",), ("goal", False), ("configure",)]
    assert writes.count(("disable",)) == 1


def test_tracking_failure_disables_torque(monkeypatch):
    robot, mapping, writes = setup_robot(monkeypatch, follow=False)
    with pytest.raises(RuntimeError, match="tracking error"):
        h.home(robot, mapping, np.full(6, 0.5), execute=True)
    assert writes[-1] == ("disable",)
    assert writes.count(("disable",)) == 2


def test_encoder_endpoint_tolerance_does_not_allow_large_range_errors(monkeypatch):
    robot, mapping, writes = setup_robot(monkeypatch)
    robot.bus.sync_read = lambda name, normalize=True: dict.fromkeys(h.JOINTS, 0.0 if normalize else 4097)
    h.home(robot, mapping, np.full(6, 0.1))
    robot.bus.sync_read = lambda name, normalize=True: dict.fromkeys(h.JOINTS, 0.0 if normalize else 4100)
    with pytest.raises(ValueError, match="encoder"):
        h.home(robot, mapping, np.full(6, 0.1), execute=True)
    assert writes == []


def test_settling_timeout_reports_joint_error_and_holds_measured_pose(monkeypatch, tmp_path):
    robot, mapping, writes = setup_robot(monkeypatch)
    original_write = robot.bus.sync_write
    commands = []

    def lagging_write(name, values, normalize=True):
        commands.append(values.copy())
        if normalize:
            values = values.copy()
            values["shoulder_lift"] -= 4.0  # 0.04 rad steady tracking offset, above 2° but below the fault limit.
        original_write(name, values, normalize)

    robot.bus.sync_write = lagging_write
    report_path = tmp_path / "homing.json"
    with pytest.raises(RuntimeError, match="home is NOT confirmed"):
        h.home(robot, mapping, np.full(6, 0.1), execute=True, report_path=report_path)
    report = json.loads(report_path.read_text())
    assert report["status"] == "not_home_holding_measured_pose"
    assert report["error_deg"][1] == pytest.approx(np.degrees(0.04))
    assert commands[-1]["shoulder_lift"] == pytest.approx(6.0)
    assert writes.count(("disable",)) == 1  # Only initial configuration, no dropping the arm at timeout.


def test_tracking_fault_report_does_not_claim_hold(monkeypatch, tmp_path):
    robot, mapping, writes = setup_robot(monkeypatch, follow=False)
    report_path = tmp_path / "homing.json"
    with pytest.raises(RuntimeError, match="tracking error"):
        h.home(robot, mapping, np.full(6, 0.5), execute=True, report_path=report_path)
    assert json.loads(report_path.read_text())["status"] == "failed_released"
    assert writes[-1] == ("disable",)
