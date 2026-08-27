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
    """Return cap-up vial alignment in [0, 1]."""
    axis = quat_w.new_tensor((0.0, 0.0, 1.0)).expand(quat_w.shape[0], -1)
    return quat_rotate_xyzw(quat_w, axis)[..., 2].clamp(0.0, 1.0)


def cylinder_lowest_offset(axis_z: torch.Tensor, axial_min: float, axial_max: float, radius: float) -> torch.Tensor:
    """Return the lowest point of an oriented finite cylinder relative to its root."""
    axis_z = axis_z.clamp(-1.0, 1.0)
    axial = torch.where(axis_z >= 0.0, axial_min * axis_z, axial_max * axis_z)
    radial = -radius * torch.sqrt((1.0 - axis_z.square()).clamp_min(0.0))
    return axial + radial


def symmetric_axial_keypoint_error(
    position: torch.Tensor,
    axis: torch.Tensor,
    target_position: torch.Tensor,
    target_axis: torch.Tensor,
    axial_min: float,
    axial_max: float,
) -> torch.Tensor:
    """Return RMS center/end-point error for an object with unsigned axial symmetry.

    ``position`` is the authored object root, which need not be its geometric
    center. Comparing the physical center and both axial endpoints preserves
    that offset while the minimum endpoint assignment makes the axis unsigned.
    """
    midpoint = 0.5 * (axial_min + axial_max)
    center = position + midpoint * axis
    lower = position + axial_min * axis
    upper = position + axial_max * axis
    target_center = target_position + midpoint * target_axis
    target_lower = target_position + axial_min * target_axis
    target_upper = target_position + axial_max * target_axis

    center_error = torch.sum(torch.square(center - target_center), dim=-1)
    direct = center_error + torch.sum(torch.square(lower - target_lower), dim=-1)
    direct += torch.sum(torch.square(upper - target_upper), dim=-1)
    swapped = center_error + torch.sum(torch.square(lower - target_upper), dim=-1)
    swapped += torch.sum(torch.square(upper - target_lower), dim=-1)
    return torch.sqrt(torch.minimum(direct, swapped) / 3.0)
