"""Pure tensor geometry used by the environment and unit tests."""

import torch


def quat_conjugate_xyzw(quat: torch.Tensor) -> torch.Tensor:
    """Return the conjugate of an XYZW quaternion."""
    return torch.cat((-quat[..., :3], quat[..., 3:]), dim=-1)


def quat_rotate_xyzw(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by unit XYZW quaternions."""
    xyz = quat[..., :3]
    return vector + 2.0 * (
        quat[..., 3:] * torch.cross(xyz, vector, dim=-1) + torch.cross(xyz, torch.cross(xyz, vector, dim=-1), dim=-1)
    )


def rack_local_position(point_w: torch.Tensor, rack_pos_w: torch.Tensor, rack_quat_w: torch.Tensor) -> torch.Tensor:
    """Transform world points into the rack frame."""
    return quat_rotate_xyzw(quat_conjugate_xyzw(rack_quat_w), point_w - rack_pos_w)


def inside_bounds(point: torch.Tensor, lower: tuple[float, ...], upper: tuple[float, ...]) -> torch.Tensor:
    """Return a mask for points inside inclusive axis-aligned bounds."""
    lo = point.new_tensor(lower)
    hi = point.new_tensor(upper)
    return ((point >= lo) & (point <= hi)).all(dim=-1)


def vertical_alignment(quat_w: torch.Tensor) -> torch.Tensor:
    """Return the vial local-Z/world-Z alignment, ignoring which end points upward."""
    axis = quat_w.new_tensor((0.0, 0.0, 1.0)).expand(quat_w.shape[0], -1)
    return quat_rotate_xyzw(quat_w, axis)[..., 2].abs()
