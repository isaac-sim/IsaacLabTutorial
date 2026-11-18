"""Gaze action of the active-gaze task, its reset, and the teacher's gaze labels.

Gaze angles are (tilt, pan) [deg] from the scene camera's optical axis: positive tilt looks up and positive pan looks
left, as for :meth:`isaaclab_fovi.sensors.FoveatedCamera.set_gaze_angles`. The policy sees and moves the gaze in field
radii, half the retina's field of view. Camera-frame directions use the OpenCV convention: x right, y down, z forward.
"""

from __future__ import annotations

import math

import torch
from isaaclab_fovi.sensors.calibrated_geometry import resolve_camera_model, sampling_geometry

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_tutorial.tasks.place_vial.mdp.geometry import RACK_HOLE_CENTERS
from isaaclab_tutorial.tasks.place_vial.mdp.progress import placement_progress
from isaaclab_tutorial.tasks.place_vial.mdp.task import RACK_RIM_HEIGHT, VIAL_AXIS_MAX, VIAL_AXIS_MIN

from .goals import CAP_LABEL_POSITION, RACK_LABEL_OFFSETS, target_holes
from .sim2real import mount_randomization

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


def clamp_gaze(angles: torch.Tensor, limit: float) -> torch.Tensor:
    """Scale gaze angles back onto the circle of radius ``limit`` when they leave it."""
    return angles * (limit / angles.norm(dim=-1, keepdim=True).clamp_min(limit))


def random_gaze(count: int, limit: float, device=None) -> torch.Tensor:
    """Gaze angles drawn uniformly over the disc of radius ``limit``, shape (count, 2)."""
    radius = limit * torch.rand(count, device=device).sqrt()
    angle = 2 * math.pi * torch.rand(count, device=device)
    return torch.stack((radius * angle.sin(), radius * angle.cos()), dim=-1)


def camera_directions(points: torch.Tensor, position: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Camera-frame unit directions to world points, for a camera at ``position`` with OpenGL ``rotation`` (xyzw)."""
    directions = quat_apply_inverse(rotation.expand(len(points), -1), points - position)
    return torch.nn.functional.normalize(directions * directions.new_tensor((1, -1, -1)), dim=-1)


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


class GazeAction(ActionTerm):
    """Moves the scene camera's gaze by bounded increments in (tilt, pan) order.

    Actions, limits and the observed gaze are in field radii. An action of 1 changes an angle by ``step_size`` and the
    gaze stays within ``gaze_limit`` of the optical axis. At reset the gaze starts at one of :data:`RESET_MODES`, drawn
    uniformly, or on the optical axis when ``mixed_resets`` is off.
    """

    cfg: GazeActionCfg

    def __init__(self, cfg: GazeActionCfg, env):
        super().__init__(cfg, env)
        self.camera = env.scene[cfg.camera_name]
        self.camera_model = resolve_camera_model(self.camera.cfg)
        self.field_radius_deg = sampling_geometry(self.camera.cfg)[0] / 2
        """Half the retina's field of view [deg]."""
        self.base_rotation = torch.tensor(self.camera.cfg.offset.rot, device=self.device)
        self._raw = torch.zeros(self.num_envs, 2, device=self.device)
        self._gaze = torch.zeros_like(self._raw)
        self._velocity = torch.zeros_like(self._raw)
        self.reset_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        """Index into :data:`RESET_MODES` of each environment's last reset."""

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._gaze

    @property
    def gaze(self) -> torch.Tensor:
        """Current gaze in field radii, shape (num_envs, 2)."""
        return self._gaze

    @property
    def angles_deg(self) -> torch.Tensor:
        """Current gaze angles [deg], shape (num_envs, 2)."""
        return self._gaze * self.field_radius_deg

    def process_actions(self, actions: torch.Tensor):
        self._raw.copy_(actions)
        previous = self._gaze.clone()
        self._gaze.copy_(clamp_gaze(self._gaze + actions.clamp(-1, 1) * self.cfg.step_size, self.cfg.gaze_limit))
        self._velocity.copy_((self._gaze - previous) / self._env.step_dt)
        self.camera.set_gaze_angles(self.angles_deg)

    def apply_actions(self):
        pass

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device)[slice(None) if env_ids is None else env_ids]
        self._raw[ids] = 0
        self._velocity[ids] = 0
        self._gaze[ids] = 0
        self.reset_mode[ids] = 0
        if self.cfg.mixed_resets and len(ids):
            self._reset_mixed(ids)
        self.camera.set_gaze_angles(self.angles_deg)

    def _reset_mixed(self, ids: torch.Tensor):
        """Start each gaze at one of :data:`RESET_MODES` for this episode's vial and rack."""
        env = self._env
        if not hasattr(env, "_so101_reset_vial_pose"):
            return
        origins = env.scene.env_origins[ids]
        vial, rack = env._so101_reset_vial_pose[ids].clone(), env._so101_reset_rack_pose[ids].clone()
        vial[:, :3] += origins
        rack[:, :3] += origins
        vial_point, goal_point = gaze_targets(env, vial, rack, ids)
        modes = torch.randint(len(RESET_MODES), (len(ids),), device=self.device)
        point = torch.where(((modes == 2) | (modes == 4))[:, None], goal_point, vial_point)
        direction = self.directions(point, ids)
        direction = torch.where((modes >= 3)[:, None], nearby_directions(direction), direction)
        aligned = direction_to_angles(direction) / self.field_radius_deg
        modes = torch.where(self.visible(direction) & (aligned.norm(dim=-1) <= self.cfg.gaze_limit), modes, 0)
        # A random start lies inside the physical image too; the optical axis is the fallback.
        random = random_gaze(len(ids), self.cfg.gaze_limit, self.device)
        random = torch.where(self.visible(angles_to_direction(random * self.field_radius_deg))[:, None], random, 0)
        self._gaze[ids] = torch.where((modes == 0)[:, None], random, aligned)
        self.reset_mode[ids] = modes

    def directions(self, points: torch.Tensor, env_ids=slice(None)) -> torch.Tensor:
        """Camera-frame unit directions from the camera's mount to world points."""
        mount = mount_randomization(self._env)
        rotation = self.base_rotation if mount is None else mount.rotation[env_ids]
        return camera_directions(points, self.camera.data.pos_w.torch[env_ids], rotation)

    def visible(self, direction: torch.Tensor) -> torch.Tensor:
        """Whether camera-frame directions fall inside the physical camera's image."""
        return self.camera_model.project(direction)[1]

    def teacher_action(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The action that moves the gaze toward world points, and whether each point is a valid target.

        A point is valid when the physical camera sees it and the gaze can reach it within the limit.
        """
        direction = self.directions(points)
        desired = direction_to_angles(direction) / self.field_radius_deg
        valid = self.visible(direction) & (desired.norm(dim=-1) <= self.cfg.gaze_limit)
        action = ((clamp_gaze(desired, self.cfg.gaze_limit) - self._gaze) / self.cfg.step_size).clamp(-1, 1)
        return torch.where(valid[:, None], action, 0.0), valid

    def fixation_error_deg(self, points: torch.Tensor) -> torch.Tensor:
        """Angle [deg] between the gaze and the direction to world points."""
        direction = self.directions(points)
        gaze = angles_to_direction(self.angles_deg)
        # atan2 stays accurate near perfect alignment, unlike acos of the dot product.
        return torch.atan2(torch.cross(gaze, direction, dim=-1).norm(dim=-1), (gaze * direction).sum(-1)).rad2deg()

    def proprioception(self) -> torch.Tensor:
        """Gaze and its velocity, in field radii and field radii per second, shape (num_envs, 4)."""
        return torch.cat((self._gaze, self._velocity), dim=-1)


@configclass
class GazeActionCfg(ActionTermCfg):
    """Configuration of :class:`GazeAction`."""

    class_type: type = GazeAction
    asset_name: str = "robot"
    camera_name: str = "eye"
    """Scene name of the foveated camera whose gaze the action moves."""
    step_size: float = 0.1
    """Change of each gaze coordinate per step at an action of 1, in field radii."""
    gaze_limit: float = 1.0
    """Largest distance of the gaze from the camera's optical axis, in field radii."""
    mixed_resets: bool = True
    """Start episodes at a gaze drawn from :data:`RESET_MODES`; otherwise on the optical axis."""


def teacher_target(env) -> torch.Tensor:
    """World point the teacher looks at: the vial until it is grasped, then the goal."""
    vial, rack = (
        torch.cat((env.scene[name].data.root_pos_w.torch, env.scene[name].data.root_quat_w.torch), dim=-1)
        for name in ("vial", "rack")
    )
    vial_point, goal_point = gaze_targets(env, vial, rack)
    return torch.where(placement_progress(env).grasped[:, None], goal_point, vial_point)


def oracle_labels(env) -> torch.Tensor:
    """The teacher's gaze action and its validity, shape (num_envs, 3). A training label, never a policy input."""
    gaze: GazeAction = env.action_manager.get_term("gaze")
    point = teacher_target(env)
    action, valid = gaze.teacher_action(point)
    error = gaze.fixation_error_deg(point)
    log = env.extras.setdefault("log", {})
    log["Gaze/target_error_deg"] = torch.where(valid, error, 0).sum() / valid.sum().clamp_min(1)
    log["Gaze/within_1deg"] = (valid & (error <= 1)).float().mean()
    log["Gaze/goal_target_fraction"] = placement_progress(env).grasped.float().mean()
    log["Gaze/valid_target_fraction"] = valid.float().mean()
    return torch.cat((action, valid[:, None].float()), dim=-1)
