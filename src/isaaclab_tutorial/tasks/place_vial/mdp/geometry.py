"""Pure tensor geometry used by the environment and unit tests."""

from functools import lru_cache

import torch

# Rack-local centers of the four open cells in workshop/rack.usda, in top_01..top_04 order.
# The original opening remains the rack-frame origin; the collision lattice has a 60 mm pitch.
RACK_HOLE_CENTERS = ((0.0, 0.0, 0.0), (0.060, 0.0, 0.0), (0.060, 0.060, 0.0), (0.0, 0.060, 0.0))


@lru_cache(maxsize=64)
def _constant(values: tuple, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Constants are shared read-only. Avoid retaining inference tensors across callers.
    with torch.inference_mode(False):
        return torch.tensor(values, device=device, dtype=dtype)


def constant_like(reference: torch.Tensor, values: tuple) -> torch.Tensor:
    """Return a read-only constant on the reference's concrete device and dtype."""
    return _constant(values, reference.device, reference.dtype)


def hole_relative_positions(rack_position: torch.Tensor) -> torch.Tensor:
    """Return offsets to all four openings, preserving rack-local height: (..., 4, 3)."""
    return rack_position.unsqueeze(-2) - constant_like(rack_position, RACK_HOLE_CENTERS)


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
    lo = constant_like(point, lower)
    hi = constant_like(point, upper)
    return ((point >= lo) & (point <= hi)).all(dim=-1)


def vertical_alignment(quat_w: torch.Tensor) -> torch.Tensor:
    """Return cap-up vial alignment in [0, 1]."""
    axis = constant_like(quat_w, (0.0, 0.0, 1.0)).expand(quat_w.shape[0], -1)
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


def tabletop_vial_overlaps_rack(
    vial_pose: torch.Tensor, rack_pose: torch.Tensor, clearance: float = 0.0
) -> torch.Tensor:
    """Conservative XY overlap of a resting vial with the rack's lower deck.

    The horizontal body and cap are tested separately against the deck with the
    separating-axis test. A small tilt expands their projected axial extents.
    Conservative bounds cover the measured cylinders and rounded bottom, and the rack base.
    This is a tabletop reset check, not an insertion/collision predicate.
    """
    position = rack_local_position(vial_pose[:, :3], rack_pose[:, :3], rack_pose[:, 3:])
    world_axis = quat_rotate_xyzw(
        vial_pose[:, 3:], constant_like(vial_pose, (0.0, 0.0, 1.0)).expand(len(vial_pose), -1)
    )
    axis = quat_rotate_xyzw(quat_conjugate_xyzw(rack_pose[:, 3:]), world_axis)
    length = axis[:, :2].norm(dim=-1).clamp_min(1e-8)
    along = axis[:, :2] / length[:, None]
    across = torch.stack((-along[:, 1], along[:, 0]), dim=-1)
    deck_center = constant_like(position, (0.0301682871, 0.0301424648))
    deck_half = constant_like(position, (0.06 + clearance, 0.06 + clearance))
    overlaps = torch.zeros(len(position), device=position.device, dtype=torch.bool)
    for lower, upper, radius in ((-0.01725, 0.086612839, 0.0147), (0.08676, 0.09953, 0.01795)):
        delta = position[:, :2] + axis[:, :2] * ((lower + upper) / 2) - deck_center
        half_length = (upper - lower) / 2 * length + radius * axis[:, 2].abs()
        xy_overlap = (delta.abs() <= deck_half + along.abs() * half_length[:, None] + across.abs() * radius).all(-1)
        axial_overlap = (delta * along).sum(-1).abs() <= half_length + (deck_half * along.abs()).sum(-1)
        radial_overlap = (delta * across).sum(-1).abs() <= radius + (deck_half * across.abs()).sum(-1)
        overlaps |= xy_overlap & axial_overlap & radial_overlap
    return overlaps
