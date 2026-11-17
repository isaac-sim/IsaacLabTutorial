"""Typed camera randomization settings for the SO-101 transfer preset."""

import math

from isaaclab.utils import configclass


@configclass
class CameraRandomizationCfg:
    position_range: float = 0.03
    rotation_deg: float = 7.0
    nominal_fraction: float = 0.2
    ramp_steps: int = 30000

    def validate_config(self):
        settings(self)


DEFAULTS = CameraRandomizationCfg().to_dict()


def settings(overrides=None):
    if isinstance(overrides, CameraRandomizationCfg):
        overrides = overrides.to_dict()
    result = DEFAULTS | (overrides or {})
    if set(result) != set(DEFAULTS):
        raise ValueError("Unknown SO-101 sim2real setting")
    if any(not math.isfinite(v) or v < 0 for v in result.values()):
        raise ValueError("Sim2real ranges must be finite and nonnegative")
    if result["nominal_fraction"] > 1:
        raise ValueError("Invalid nominal sampling fraction")
    return result
