"""Deployment geometry, action-state transitions, and delayed-result handling."""

import math
import time

import numpy as np
import pytest
import torch

from isaaclab_tutorial.active_gaze.deployment.control import (
    FORMAT,
    GAZE,
    JOINTS,
    PROPRIOCEPTION,
    PROPRIOCEPTION_SIZE,
    PolicyState,
    advance_policy_deadline,
)
from isaaclab_tutorial.active_gaze.deployment.retina import SoftwareRetina
from isaaclab_tutorial.active_gaze.deployment.runtime import InferenceWorker


def contract():
    return dict(
        format=FORMAT,
        joint_order=list(JOINTS),
        joint_units="radians",
        proprioception=list(PROPRIOCEPTION),
        policy_hz=30,
        relative_target_hz=120,
        action_scale=[0.033] * 5 + [0.02],
        gaze=dict(step_size=0.1, limit=1.0, field_radius_deg=39.0),
    )


def pinhole(fov_deg=78.0, size=640):
    """An ideal square pinhole camera's model."""
    focal = size / 2 / math.tan(math.radians(fov_deg / 2))
    center = (size - 1) / 2
    return dict(model="pinhole", image_size=(size, size), intrinsics=(focal, focal, center, center))


def test_state_moves_the_gaze_in_field_radii_within_the_simulators_radial_limit():
    state = PolicyState(contract())
    state.reset([0.7, -0.7])
    state.commit([2, 0, 0, 0, 0, -2, 1, -1])
    # The step leaves the unit circle, so the gaze is scaled back onto it, as in simulation.
    np.testing.assert_allclose(state.gaze, [0.5**0.5, -(0.5**0.5)], atol=1e-6)
    np.testing.assert_allclose(state.gaze_velocity, (state.gaze - [0.7, -0.7]) * 30, atol=1e-5)
    np.testing.assert_allclose(state.angles_deg, state.gaze * 39.0)
    observed = state.observation(np.ones(6), np.ones(6) * 13, np.ones(6) * 2)
    assert observed.shape == (PROPRIOCEPTION_SIZE,)
    np.testing.assert_array_equal(observed[6:12], [13] * 6)
    np.testing.assert_array_equal(observed[18:26], state.action)
    np.testing.assert_array_equal(observed[GAZE], state.gaze)
    np.testing.assert_array_equal(observed[28:30], state.gaze_velocity)
    limits = np.array([[-4, 4]] * 6)
    first = state.target(np.zeros(6), limits)
    second = state.target(np.ones(6), limits)
    np.testing.assert_allclose(second - first, np.ones(6))
    state.reset()
    np.testing.assert_array_equal(
        state.observation(np.zeros(6), np.zeros(6), np.zeros(6)), np.zeros(PROPRIOCEPTION_SIZE)
    )
    with pytest.raises(ValueError):
        state.commit([float("nan")] * 8)
    with pytest.raises(ValueError):
        state.reset([0.8, 0.8])


def test_policy_clock_preserves_rate_with_bus_jitter_and_skips_missed_periods():
    deadline, ticks = 0.0, []
    for index in range(1200):
        now = index / 120 + (index % 7) * 0.00003
        if now >= deadline:
            deadline, missed = advance_policy_deadline(deadline, now)
            assert missed == 0
            ticks.append(now)
    assert len(ticks) == 300
    assert deadline == pytest.approx(10)
    deadline, missed = advance_policy_deadline(1, 1.11)
    assert deadline > 1.11
    assert missed == 3


def test_software_retina_is_upright_follows_the_gaze_and_blanks_what_the_camera_misses():
    retina = SoftwareRetina(pinhole(), 78.0, 0.591, 128)
    yy, xx = torch.meshgrid(torch.arange(640), torch.arange(640), indexing="ij")
    rgb = torch.stack((xx / 3, yy / 3, torch.full_like(xx, 100)), -1).to(torch.uint8)[None]
    # Gaze (tilt, pan) [deg]: positive tilt looks up, positive pan looks left.
    gaze = torch.tensor([[0.0, 0.0], [17.0, -23.0]])
    grid, valid = retina.grid(gaze)
    image = retina(rgb.expand(2, -1, -1, -1), gaze)
    assert image.shape == (2, 128, 128, 3)
    # Upright public raster: right means larger source x, down means larger source y.
    assert image[0, 64, 80, 0] > image[0, 64, 48, 0]
    assert image[0, 80, 64, 1] > image[0, 48, 64, 1]
    # Looking up and right moves the fixation to smaller source y and larger source x.
    assert image[1, 64, 64, 0] > image[0, 64, 64, 0]
    assert image[1, 64, 64, 1] < image[0, 64, 64, 1]
    # Looking straight ahead, the square 78 degree camera sees the whole circular 78 degree retina; looking away, it
    # cuts part of it off.
    assert torch.equal(valid[0], retina.valid[0])
    assert valid[1].sum() < valid[0].sum()
    assert (image[~valid.expand_as(image)] == 0).all()
    assert torch.isfinite(grid).all()


def test_worker_does_not_accept_stale_or_overlapping_results():
    class Runtime:
        def infer(self, proprio, rgb):
            return np.arange(8, dtype=np.float32)

    worker = InferenceWorker(Runtime())
    try:
        worker.submit(np.zeros(PROPRIOCEPTION_SIZE), np.zeros((2, 2, 3)), time.monotonic() - 1)
        with pytest.raises(RuntimeError, match="already"):
            worker.submit(np.zeros(PROPRIOCEPTION_SIZE), np.zeros((2, 2, 3)), time.monotonic())
        with pytest.raises(RuntimeError, match="stale"):
            worker.poll()
    finally:
        worker.close()


@pytest.mark.parametrize("execute,fail", [(False, False), (True, False), (True, True)])
def test_hardware_adapter_read_only_startup_and_error_shutdown(tmp_path, monkeypatch, execute, fail):
    import hashlib
    import json
    import sys
    from types import ModuleType, SimpleNamespace

    from isaaclab_tutorial.active_gaze.deployment import run

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(run.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(run.time, "sleep", lambda delay: setattr(clock, "now", clock.now + delay))
    calls, observations = [], []
    position = np.full(6, 0.2)

    class Bus:
        is_connected = False

        def connect(self):
            self.is_connected = True

        def sync_read(self, name, **kwargs):
            clock.now += 0.001
            return dict(zip(JOINTS, position.tolist(), strict=True))

        def sync_write(self, name, value, **kwargs):
            calls.append(("write", name, value))

        def disable_torque(self):
            calls.append(("disable",))

        def disconnect(self, disable_torque):
            calls.append(("disconnect", disable_torque))
            self.is_connected = False

    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}")

    class Robot:
        def __init__(self, cfg):
            self.bus = Bus()
            self.is_calibrated = True
            self.calibration_fpath = calibration
            self.calibration = {n: SimpleNamespace(range_min=0, range_max=4095) for n in JOINTS}

        def configure(self):
            calls.append(("configure",))

        def send_action(self, action):
            calls.append(("action", action))
            return action

    module = ModuleType("lerobot.robots.so_follower")
    module.SO101Follower, module.SO101FollowerConfig = Robot, lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "lerobot.robots.so_follower", module)

    class Runtime:
        def __init__(self, *args):
            self.contract = dict(contract(), camera_model={"image_size": [2, 2]}, calibrated_capture=True)

        def infer(self, *args):
            clock.now += 2  # Long warmup while someone moves the arm.
            position[:] = 0.3
            return np.zeros(8)

    class Camera:
        latest, error = True, None

        def __init__(self, *args):
            pass

        def timestamped_frame(self):
            return clock.now, np.zeros((2, 2, 3), dtype=np.uint8)

        def close(self):
            calls.append(("camera_close",))

    class Worker:
        future = None

        def __init__(self, *args):
            pass

        def poll(self):
            if self.future is None:
                return None
            if fail:
                raise RuntimeError("stale inference")
            self.future = None
            return np.zeros(8)

        def submit(self, proprio, image, stamp):
            observations.append(proprio.copy())
            self.future, self.started = True, stamp

        def close(self):
            calls.append(("worker_close",))

    monkeypatch.setattr(run, "PolicyRuntime", Runtime)
    monkeypatch.setattr(run, "SceneCamera", Camera)
    monkeypatch.setattr(run, "InferenceWorker", Worker)
    mapping = tmp_path / "map.json"
    mapping.write_text(
        json.dumps(
            dict(
                units="lerobot_degrees_and_gripper_percent",
                verified=True,
                calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
                joints={n: dict(scale=1, offset=0, limits=[-4, 4]) for n in JOINTS},
            )
        )
    )
    start = tmp_path / "start.json"
    start.write_text(json.dumps(dict(position=[0.2] * 6, tolerance_rad=0.5)))
    argv = [
        "run",
        "--bundle",
        str(tmp_path),
        "--joint-map",
        str(mapping),
        "--port",
        "fake",
        "--duration",
        "0.15",
    ]
    if execute:
        argv += ["--execute", "--start-pose", str(start)]
    monkeypatch.setattr(sys, "argv", argv)
    if fail:
        with pytest.raises(RuntimeError, match="stale"):
            run.main()
    else:
        run.main()
    np.testing.assert_array_equal(observations[0][6:12], np.zeros(6))
    if execute:
        np.testing.assert_allclose(observations[0][12:18], [0.3] * 6)
        names = [call[0] for call in calls]
        assert names.index("write") < names.index("configure") < names.index("disconnect")
        seeded = next(call[2] for call in calls if call[0] == "write")
        assert list(seeded.values()) == pytest.approx([0.3] * 6)
        assert ("disconnect", True) in calls
    else:
        assert not any(call[0] in ("write", "disable", "configure", "action") for call in calls)
        assert ("disconnect", False) in calls
    assert next(i for i, call in enumerate(calls) if call[0] == "disconnect") < calls.index(("worker_close",))


def test_export_rejects_chunked_actor_before_loading_weights(tmp_path):
    import json

    from isaaclab_tutorial.active_gaze.deployment.model import load_policy

    run = dict(policy="dino_language", retina=dict(resolution=128), action_chunk_size=3)
    (tmp_path / "run.json").write_text(json.dumps(run))
    (tmp_path / "agent.json").write_text(json.dumps({"obs_groups": {"actor": ["proprio", "vision", "instruction"]}}))
    with pytest.raises(ValueError, match="single-step actions"):
        load_policy(tmp_path / "model.pt", pinhole(), "cpu")
