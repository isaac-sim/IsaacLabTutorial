"""What the active-gaze task adds to :mod:`isaaclab_fovi.mdp`'s gaze action: where the gaze starts an episode, the
world point its teacher looks at, and the teacher's task logs.

The action, its observation and the teacher's labels are isaaclab_fovi's. Gaze angles are (tilt, pan) [deg] from the
scene camera's optical axis: positive tilt looks up and positive pan looks left. The policy sees and moves the gaze in
field radii, half the retina's field of view. Camera-frame directions use the ROS (OpenCV) convention: x right, y
down, z forward.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from isaaclab_fovi.mdp import GazeAction, GazeActionCfg, gaze_teacher_labels

from isaaclab_tutorial.tasks.place_vial.mdp.geometry import RACK_HOLE_CENTERS
from isaaclab_tutorial.tasks.place_vial.mdp.progress import placement_progress
from isaaclab_tutorial.tasks.place_vial.mdp.task import RACK_RIM_HEIGHT, VIAL_AXIS_MAX, VIAL_AXIS_MIN

from .goals import CAP_LABEL_POSITION, RACK_LABEL_OFFSETS, target_holes

RESET_MODES = ("random", "object", "goal", "near_object", "near_goal")
"""Where the gaze starts an episode: anywhere in the camera's view, at the vial or the goal, or 2 to 5 degrees off
either."""


def angles_to_direction(angles_deg: torch.Tensor) -> torch.Tensor:
    """Camera-frame unit directions of gaze angles [deg], shape (..., 2) to (..., 3)."""
    tilt, pan = angles_deg.deg2rad().unbind(-1)
    return torch.stack((-tilt.cos() * pan.sin(), -tilt.sin(), tilt.cos() * pan.cos()), -1)


def direction_to_angles(direction: torch.Tensor) -> torch.Tensor:
    """Gaze angles [deg] of camera-frame directions, shape (..., 3) to (..., 2)."""
    x, y, z = direction.unbind(-1)
    return torch.stack((-torch.atan2(y, torch.sqrt(x.square() + z.square())), -torch.atan2(x, z)), -1).rad2deg()


def nearby_directions(direction: torch.Tensor) -> torch.Tensor:
    """Directions 2 to 5 degrees from each given direction, in a uniformly drawn azimuth."""
    reference = direction.new_tensor((0, 1, 0)).expand_as(direction)
    tangent = torch.nn.functional.normalize(torch.cross(direction, reference, dim=-1), dim=-1)
    bitangent = torch.cross(direction, tangent, dim=-1)
    theta = torch.empty(len(direction), 1, device=direction.device).uniform_(math.radians(2), math.radians(5))
    phi = torch.rand_like(theta) * (2 * math.pi)
    return theta.cos() * direction + theta.sin() * (phi.cos() * tangent + phi.sin() * bitangent)


def target_centers(vial_pose, rack_pose, holes=None, object_offset=None, goal_offsets=None):
    """World points of the vial and of the goal opening.

    The goal is the given hole of each rack, or the opening nearest the vial. Offsets move the points from the vial's
    center and the openings' rim centers, for example onto the number labels.
    """
    object_center = vial_pose[:, :3] + quat_apply(
        vial_pose[:, 3:],
        vial_pose.new_tensor(object_offset or (0, 0, (VIAL_AXIS_MIN + VIAL_AXIS_MAX) / 2)).expand(len(vial_pose), -1),
    )
    openings = vial_pose.new_tensor(RACK_HOLE_CENTERS).expand(len(vial_pose), -1, -1).clone()
    openings[..., 2] = RACK_RIM_HEIGHT
    if goal_offsets is not None:
        openings[..., :2] += openings.new_tensor(goal_offsets)
        openings[..., 2] += 0.0002
    world = rack_pose[:, None, :3] + quat_apply(rack_pose[:, None, 3:].expand(-1, 4, -1), openings)
    if holes is None:
        holes = (world[..., :2] - vial_pose[:, None, :2]).square().sum(-1).argmin(-1)
    return object_center, world[torch.arange(len(vial_pose), device=vial_pose.device), holes]


def gaze_targets(env, vial_pose: torch.Tensor, rack_pose: torch.Tensor, env_ids=slice(None)):
    """World points the gaze should find: the vial and the goal, on their number labels when the vial is numbered."""
    holes = target_holes(env)
    if holes is None:
        return target_centers(vial_pose, rack_pose)
    return target_centers(vial_pose, rack_pose, holes[env_ids], CAP_LABEL_POSITION, RACK_LABEL_OFFSETS)


def teacher_target(env) -> torch.Tensor:
    """World point the teacher looks at: the vial until it is grasped, then the goal."""
    vial, rack = (
        torch.cat((env.scene[name].data.root_pos_w.torch, env.scene[name].data.root_quat_w.torch), dim=-1)
        for name in ("vial", "rack")
    )
    vial_point, goal_point = gaze_targets(env, vial, rack)
    return torch.where(placement_progress(env).grasped[:, None], goal_point, vial_point)


def target_directions(gaze: GazeAction, points: torch.Tensor, env_ids=slice(None)) -> torch.Tensor:
    """Camera-frame unit directions from the gaze's camera, as it stands now, to world points.

    The scene camera has no image roll, so its frame is the upright frame in which gazes are measured.
    """
    mounts = gaze.teacher_mounts().torch[env_ids]
    # Reading the mounts refreshed the camera's position without capturing an image.
    position = gaze.camera.mount_pos_w.torch[env_ids]
    return torch.nn.functional.normalize(quat_apply_inverse(mounts, points - position), dim=-1)


def fixation_error_deg(gaze: GazeAction, points: torch.Tensor) -> torch.Tensor:
    """Angle [deg] between the gaze and the direction to world points."""
    direction = target_directions(gaze, points)
    looking = angles_to_direction(gaze.angles_deg)
    # atan2 stays accurate near perfect alignment, unlike acos of the dot product.
    return torch.atan2(torch.cross(looking, direction, dim=-1).norm(dim=-1), (looking * direction).sum(-1)).rad2deg()


def mixed_reset_gaze(
    vial_direction: torch.Tensor, goal_direction: torch.Tensor, visible, field_radius_deg: float, gaze_limit: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw where each gaze starts an episode.

    Args:
        vial_direction: Camera-frame directions to each environment's vial, shape (N, 3).
        goal_direction: Camera-frame directions to each environment's goal, shape (N, 3).
        visible: Whether camera-frame directions fall inside the physical camera's image.
        field_radius_deg: Half the retina's field of view [deg].
        gaze_limit: Largest distance of the gaze from the optical axis, in field radii.

    Returns:
        Each environment's index into :data:`RESET_MODES`, drawn uniformly, and its start gaze in field radii, shape
        (N, 2). A start the camera does not see or the gaze cannot reach becomes mode 0, whose gaze is not used.
    """
    modes = torch.randint(len(RESET_MODES), (len(vial_direction),), device=vial_direction.device)
    direction = torch.where(((modes == 2) | (modes == 4))[:, None], goal_direction, vial_direction)
    direction = torch.where((modes >= 3)[:, None], nearby_directions(direction), direction)
    gaze = direction_to_angles(direction) / field_radius_deg
    reachable = visible(direction) & (gaze.norm(dim=-1) <= gaze_limit)
    return torch.where(reachable, modes, 0), gaze


class MixedResetGazeAction(GazeAction):
    """isaaclab_fovi's gaze action, with episodes that start at one of :data:`RESET_MODES`, drawn uniformly.

    The ``random`` mode is isaaclab_fovi's own reset: a random gaze the camera sees with ``random_reset``, else the
    optical axis.
    """

    cfg: MixedResetGazeActionCfg

    def __init__(self, cfg: MixedResetGazeActionCfg, env):
        super().__init__(cfg, env)
        self.reset_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        """Index into :data:`RESET_MODES` of each environment's last reset."""

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = torch.arange(self.num_envs, device=self.device)[slice(None) if env_ids is None else env_ids]
        self.reset_mode[ids] = 0
        env = self._env
        # The dataset reset records each episode's vial and rack; before the first one there is nothing to look at.
        if not self.cfg.mixed_resets or not len(ids) or not hasattr(env, "_so101_reset_vial_pose"):
            return
        origins = env.scene.env_origins[ids]
        vial, rack = env._so101_reset_vial_pose[ids].clone(), env._so101_reset_rack_pose[ids].clone()
        vial[:, :3] += origins
        rack[:, :3] += origins
        directions = (target_directions(self, point, ids) for point in gaze_targets(env, vial, rack, ids))
        modes, gaze = mixed_reset_gaze(*directions, self.visible, self.field_radius_deg, self.cfg.gaze_limit)
        self.reset_mode[ids] = modes
        aimed = ids[modes > 0]
        # The camera's stored angles own the gaze, and the observed state starts there, at rest.
        angles = self.camera.gaze_angles.torch
        angles[aimed] = gaze[modes > 0] * self.field_radius_deg
        self.state[aimed, :2] = gaze[modes > 0]
        self.camera.set_gaze_angles(angles)


@configclass
class MixedResetGazeActionCfg(GazeActionCfg):
    """Configuration of :class:`MixedResetGazeAction`."""

    class_type: type = MixedResetGazeAction
    random_reset: bool = True
    mixed_resets: bool = True
    """Start episodes at a gaze drawn from :data:`RESET_MODES`; otherwise as isaaclab_fovi's gaze action does."""


class oracle_labels(gaze_teacher_labels):
    """The teacher's gaze action toward :func:`teacher_target` and its validity, shape (num_envs, 3).

    A training label, never a policy input. Next to isaaclab_fovi's logs, it logs the fraction of environments whose
    gaze is within one degree of a valid target, and the fraction whose target is the goal.
    """

    def __call__(self, env) -> torch.Tensor:
        points = teacher_target(env)
        labels = super().__call__(env, target_w=lambda _: points)
        valid = labels[:, 2] > 0.5
        error = fixation_error_deg(env.action_manager.get_term("gaze"), points)
        log = env.extras.setdefault("log", {})
        log["Gaze/within_1deg"] = (valid & (error <= 1)).float().mean()
        log["Gaze/goal_target_fraction"] = placement_progress(env).grasped.float().mean()
        return labels
