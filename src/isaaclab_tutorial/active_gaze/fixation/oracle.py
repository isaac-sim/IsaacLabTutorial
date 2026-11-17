"""Geometric pan/tilt targets in the same bounded delta space as policy actions."""

import torch


def fixation_action_target(
    direction: torch.Tensor,
    commanded: torch.Tensor,
    *,
    max_delta: float,
    angle_limit: float,
    visible: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized (tilt, pan) deltas for mount-frame target vectors.

    The neutral rig faces +Y with +Z up. Positive pan turns left and positive
    tilt turns up. Hardware targets may lie outside the *current* image; hybrid
    callers additionally supply fixed physical-sensor visibility. The desired
    joint target is bounded before converting to the executable action scale.
    """
    if max_delta <= 0 or angle_limit <= 0:
        raise ValueError("Gaze scale and joint limit must be positive.")
    x, y, z = direction.unbind(-1)
    desired = torch.stack((torch.atan2(z, torch.sqrt(x.square() + y.square())), torch.atan2(-x, y)), dim=-1)
    valid = torch.isfinite(direction).all(-1) & (direction.norm(dim=-1) > 1e-6)
    valid &= (desired.abs() <= angle_limit).all(-1)
    if visible is not None:
        valid &= visible
    action = ((desired.clamp(-angle_limit, angle_limit) - commanded) / max_delta).clamp(-1, 1)
    return torch.where(valid[:, None], action, 0.0), valid
