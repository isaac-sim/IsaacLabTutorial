"""Labels and loss of the camera mount's move, for the actor's extrinsics head."""

import torch
from torch.nn import functional as F

TRANSLATION_SCALE = 0.03
"""Translation [m] that maps to a label of 1."""
ROTATION_SCALE = 0.1
"""Rotation-matrix entry change that maps to a label of 1."""


def pose_targets(translation_delta, rotation_delta):
    identity = torch.eye(3, device=rotation_delta.device, dtype=rotation_delta.dtype)
    return torch.cat(
        (
            translation_delta / TRANSLATION_SCALE,
            ((rotation_delta[..., :2, :] - identity[:2]) / ROTATION_SCALE).flatten(-2),
        ),
        -1,
    )


def pose_loss(prediction, target):
    """Dimensionless robust translation/rotation supervision, with physical diagnostics."""
    prediction, target = prediction.float(), target.float()
    translation = F.smooth_l1_loss(prediction[..., :3], target[..., :3])
    rotation = F.smooth_l1_loss(prediction[..., 3:], target[..., 3:])
    identity = torch.eye(3, device=prediction.device)[:2]

    def matrix(value):
        rows = value[..., 3:].reshape(-1, 2, 3) * ROTATION_SCALE + identity
        first = F.normalize(rows[:, 0], dim=-1)
        second = F.normalize(rows[:, 1] - (first * rows[:, 1]).sum(-1, keepdim=True) * first, dim=-1)
        return torch.stack((first, second, torch.cross(first, second, dim=-1)), -2)

    with torch.no_grad(), torch.autocast(prediction.device.type, enabled=False):
        position_error = (prediction[..., :3] - target[..., :3]).norm(dim=-1).mean() * TRANSLATION_SCALE * 100
        relative = matrix(prediction) @ matrix(target).transpose(-1, -2)
        angle = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1).acos().rad2deg().mean()
    return translation + rotation, dict(extrinsics_translation_cm=position_error, extrinsics_rotation_deg=angle)
