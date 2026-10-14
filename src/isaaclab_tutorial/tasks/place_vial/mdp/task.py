"""Vial, rack and gripper quantities shared by the observation, reward and termination terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse

from .geometry import cylinder_lowest_offset, hole_relative_positions, rack_local_position, vertical_alignment

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Task geometry [m]. Rack quantities use the rack frame, whose origin is the first opening.
VIAL_REST_HEIGHT = 0.052
"""Root height of a vial lying on the desk."""
VIAL_AXIS_MIN, VIAL_AXIS_MAX, VIAL_RADIUS = -0.017, 0.100, 0.0177
"""Vial extent along its axis relative to the root, and its collision radius."""
VIAL_GRASP_OFFSET = (0.0, 0.0, 0.093145)
"""Cap/shoulder point the jaws close on, in the vial frame."""
FIXED_FINGERTIP_OFFSET = (-0.01289, -0.00022, -0.0986)
MOVING_FINGERTIP_OFFSET = (-0.0068, -0.0766, 0.0189)
RACK_RIM_HEIGHT = 0.073
RACK_CLEARANCE_HEIGHT = RACK_RIM_HEIGHT + 0.008
"""A vial whose lowest point is this high above the rack base has been lifted clear of the rack."""
INSERTION_RADIUS = 0.012
"""A vial tip below the rim within this distance of a hole centre is in that hole (holes are 60 mm apart)."""
UPRIGHT_ALIGNMENT = 0.90
"""Cosine of the vial axis with vertical; about 26 degrees of tilt."""
SEATED_LOWER, SEATED_UPPER = (-0.015, -0.015, 0.026), (0.015, 0.015, 0.040)
"""Hole-relative box that contains a seated vial root but not a vial resting on the rim."""
HARD_RACK_IMPACT_FORCE = 20.0
"""Light rack guidance is expected during insertion; only larger vial/rack forces [N] count as impacts."""


def contact_force(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    """Largest contact force magnitude [N] of a sensor, filtered if the sensor declares filters."""
    sensor = env.scene.sensors[sensor_name]
    forces = sensor.data.force_matrix_w if sensor.cfg.filter_prim_paths_expr else sensor.data.net_forces_w
    return torch.linalg.vector_norm(forces.torch, dim=-1).reshape(env.num_envs, -1).amax(dim=-1)


def jaw_contacts(env: ManagerBasedRLEnv, threshold: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """Whether the fixed and the moving jaw touch the vial."""
    return contact_force(env, "fixed_jaw_contact") > threshold, contact_force(env, "moving_jaw_contact") > threshold


def fingertip_positions_w(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed and moving fingertip pad centres in the world frame [m]."""
    robot = env.scene["robot"]
    pads = []
    for body, offset in (("gripper", FIXED_FINGERTIP_OFFSET), ("moving_jaw_so101_v1", MOVING_FINGERTIP_OFFSET)):
        body_id = robot.find_bodies(body)[0][0]
        pos, quat = robot.data.body_pos_w.torch[:, body_id], robot.data.body_quat_w.torch[:, body_id]
        pads.append(pos + quat_apply(quat, pos.new_tensor(offset).expand_as(pos)))
    return pads[0], pads[1]


def grasp_center_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Midpoint between the two fingertip pads in the world frame [m]."""
    fixed, moving = fingertip_positions_w(env)
    return 0.5 * (fixed + moving)


def vial_grasp_point_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The cap/shoulder point of the vial in the world frame [m]."""
    vial = env.scene["vial"]
    pos, quat = vial.data.root_pos_w.torch, vial.data.root_quat_w.torch
    return pos + quat_apply(quat, pos.new_tensor(VIAL_GRASP_OFFSET).expand_as(pos))


def vial_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Vial root height above the environment origin [m]."""
    return env.scene["vial"].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]


def vial_in_rack(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rack-local vial root position, rack-local vial axis, and the vial's lowest rack-local height."""
    vial, rack = env.scene["vial"], env.scene["rack"]
    vial_pos, vial_quat = vial.data.root_pos_w.torch, vial.data.root_quat_w.torch
    rack_pos, rack_quat = rack.data.root_pos_w.torch, rack.data.root_quat_w.torch
    position = rack_local_position(vial_pos, rack_pos, rack_quat)
    axis = quat_apply_inverse(
        rack_quat, quat_apply(vial_quat, vial_pos.new_tensor((0.0, 0.0, 1.0)).expand_as(vial_pos))
    )
    lowest = position[:, 2] + cylinder_lowest_offset(axis[:, 2], VIAL_AXIS_MIN, VIAL_AXIS_MAX, VIAL_RADIUS)
    return position, axis, lowest


def nearest_hole_distance(position: torch.Tensor) -> torch.Tensor:
    """Horizontal distance [m] from a rack-local position to the closest hole centre."""
    return torch.linalg.vector_norm(hole_relative_positions(position)[..., :2], dim=-1).amin(dim=-1)


def gripper_openness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Measured jaw opening normalized to [0, 1] over the soft joint limits."""
    robot = env.scene["robot"]
    joint_id = robot.find_joints("gripper")[0][0]
    position = robot.data.joint_pos.torch[:, joint_id]
    lower, upper = robot.data.soft_joint_pos_limits.torch[:, joint_id].unbind(-1)
    return ((position - lower) / (upper - lower)).clamp(0.0, 1.0)


def vial_upright(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Cosine of the vial axis with world vertical, clamped to [0, 1]."""
    return vertical_alignment(env.scene["vial"].data.root_quat_w.torch)
