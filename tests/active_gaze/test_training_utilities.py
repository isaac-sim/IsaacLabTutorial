"""Local training measurement contracts."""

import pytest

from isaaclab_tutorial.active_gaze.training.calibrate import valid_measurement
from isaaclab_tutorial.active_gaze.training.memory import minibatch_count


def test_minibatches_preserve_every_transition():
    assert minibatch_count(256, 32, 1024) == 8
    for envs, steps, batch in ((0, 32, 512), (128, 32, 0), (127, 32, 512), (4, 32, 256)):
        with pytest.raises(ValueError):
            minibatch_count(envs, steps, batch)


def test_measurement_rejects_incomplete_nonfinite_and_over_budget():
    measured = dict(
        complete=True,
        measured_updates=20,
        peak_conservative_bytes=70,
        total_bytes=100,
        mean_total_fps=123.0,
    )
    assert valid_measurement(measured, 20, 0.8)
    for changed in (
        dict(complete=False),
        dict(measured_updates=19),
        dict(peak_conservative_bytes=81),
        dict(peak_conservative_bytes=0),
        dict(mean_total_fps=float("nan")),
        dict(mean_total_fps=0),
        dict(error="NVML unavailable"),
    ):
        assert not valid_measurement(measured | changed, 20, 0.8)
    assert not valid_measurement(measured | dict(measured_updates=0), 0, 0.8)
