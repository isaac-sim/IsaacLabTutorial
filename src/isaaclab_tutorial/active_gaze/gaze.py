"""Physical-row-aligned gaze resets and optional grasp-gated training labels."""

import hashlib
import json
import math
from pathlib import Path

import torch
from fovi.sensing.projection import CameraModel
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_tutorial.active_gaze.fixation.actions import GazeAction, GazeActionCfg
from isaaclab_tutorial.active_gaze.fixation.oracle import fixation_action_target
from isaaclab_tutorial.tasks.place_vial.mdp.geometry import RACK_HOLE_CENTERS
from isaaclab_tutorial.tasks.place_vial.mdp.progress import placement_progress
from isaaclab_tutorial.tasks.place_vial.mdp.task import (
    RACK_RIM_HEIGHT,
    VIAL_AXIS_MAX,
    VIAL_AXIS_MIN,
)
from isaaclab_tutorial.active_gaze.reset import load_reset_dataset

RESET_MODES = ("random", "object", "goal", "near_object", "near_goal")


def oracle_target(env):
    """Switch from vial center to opening after the stock confirmed grasp.

    The milestone is restored by stock resets and stays latched through release.
    It is a training label only, never an actor input or an executed action.
    """
    poses = [
        torch.cat((env.scene[name].data.root_pos_w.torch, env.scene[name].data.root_quat_w.torch), dim=-1)
        for name in ("vial", "rack")
    ]
    object_center, goal_center = target_centers(*poses)
    if getattr(env.cfg, "goal_mode", "nearest") == "numbered":
        from .goals import CAP_LABEL_POSITION, selected_goal_center

        object_center = poses[0][:, :3] + quat_apply(
            poses[0][:, 3:], poses[0].new_tensor(CAP_LABEL_POSITION).expand(env.num_envs, -1)
        )
        goal_center = selected_goal_center(env, label=True)
    return torch.where(placement_progress(env).grasped[:, None], goal_center, object_center)


def oracle_labels(env):
    gaze = env.action_manager.get_term("gaze")
    position = oracle_target(env)
    target, valid = gaze.oracle(position)
    error = gaze.fixation_error(position).rad2deg()
    log = env.extras.setdefault("log", {})
    log["Gaze/target_error_deg"] = torch.where(valid, error, 0).sum() / valid.sum().clamp_min(1)
    log["Gaze/within_1deg"] = (valid & (error <= 1)).float().mean()
    log["Gaze/goal_target_fraction"] = placement_progress(env).grasped.float().mean()
    log["Gaze/valid_target_fraction"] = valid.float().mean()
    return torch.cat((target, valid[:, None].float()), dim=-1)


def calibrated_oracle(direction, commanded, calibration, *, max_delta, angle_limit):
    """Convert ROS camera rays to executable actions with physical-frustum masking."""
    _, visible = calibration.project(direction)
    # Shared fixation geometry expects mount right/forward/up, not ROS right/down/forward.
    mount = torch.stack((direction[:, 0], direction[:, 2], -direction[:, 1]), dim=-1)
    return fixation_action_target(mount, commanded, max_delta=max_delta, angle_limit=angle_limit, visible=visible)


def angles_to_direction(angles):
    tilt, pan = angles.unbind(-1)
    return torch.stack((-tilt.cos() * pan.sin(), -tilt.sin(), tilt.cos() * pan.cos()), -1)


def direction_to_angles(direction):
    x, y, z = direction.unbind(-1)
    return torch.stack((-torch.atan2(y, torch.sqrt(x.square() + z.square())), -torch.atan2(x, z)), -1)


def target_centers(vial_pose, rack_pose, target_holes=None, object_offset=None, goal_offsets=None):
    """Vial geometric center and the closest admissible opening's rim center."""
    object_center = vial_pose[:, :3] + quat_apply(
        vial_pose[:, 3:],
        vial_pose.new_tensor(object_offset or (0, 0, (VIAL_AXIS_MIN + VIAL_AXIS_MAX) / 2)).expand(len(vial_pose), -1),
    )
    holes = vial_pose.new_tensor(RACK_HOLE_CENTERS).expand(len(vial_pose), -1, -1).clone()
    holes[..., 2] = RACK_RIM_HEIGHT
    if goal_offsets is not None:
        holes[..., :2] += holes.new_tensor(goal_offsets)
        holes[..., 2] += 0.0002
    world_holes = rack_pose[:, None, :3] + quat_apply(rack_pose[:, None, 3:].expand(-1, 4, -1), holes)
    nearest = (world_holes[..., :2] - vial_pose[:, None, :2]).square().sum(-1).argmin(-1)
    if target_holes is not None:
        nearest = target_holes
    return object_center, world_holes[torch.arange(len(vial_pose), device=vial_pose.device), nearest]


def camera_directions(points, position, rotation):
    directions = quat_apply_inverse(rotation.expand(len(points), -1), points - position)
    directions = directions * directions.new_tensor((1, -1, -1))
    return torch.nn.functional.normalize(directions, dim=-1)


def near_directions(direction, generator, variants=16):
    """Sample a true spherical annulus, two to five degrees from the center."""
    direction = direction[:, None].expand(-1, variants, -1)
    reference = direction.new_tensor((0, 1, 0)).expand_as(direction)
    tangent = torch.nn.functional.normalize(torch.cross(direction, reference, dim=-1), dim=-1)
    bitangent = torch.cross(direction, tangent, dim=-1)
    theta = torch.empty(direction.shape[:2]).uniform_(math.radians(2), math.radians(5), generator=generator)
    phi = torch.empty_like(theta).uniform_(0, 2 * math.pi, generator=generator)
    return theta.cos()[..., None] * direction + theta.sin()[..., None] * (
        phi.cos()[..., None] * tangent + phi.sin()[..., None] * bitangent
    )


def build_reset_cache(dataset_path, camera_cfg, rack_pose, angle_limit, seed=42, numbered=False):
    """Keep stock physical rows intact and add diverse absolute gaze targets."""
    artifact = load_reset_dataset(dataset_path)
    states = artifact["states"]
    if numbered:
        from .goals import CAP_LABEL_POSITION, RACK_LABEL_OFFSETS, expand_numbered_resets

        states, _, _, _ = expand_numbered_resets(states, rack_pose)
    vial = states["vial_pose"]
    rack = torch.tensor(rack_pose, dtype=vial.dtype).expand(len(vial), -1)
    centers = target_centers(
        vial,
        rack,
        states.get("target_hole"),
        CAP_LABEL_POSITION if numbered else None,
        RACK_LABEL_OFFSETS if numbered else None,
    )
    position, rotation = (
        vial.new_tensor(camera_cfg.offset.pos),
        vial.new_tensor(camera_cfg.offset.rot),
    )
    directions = [camera_directions(center, position, rotation) for center in centers]
    generator = torch.Generator().manual_seed(seed)
    variants = 16
    random = torch.empty(len(vial), variants, 2).uniform_(-angle_limit, angle_limit, generator=generator)
    object_angles, goal_angles = [direction_to_angles(d)[:, None].expand(-1, variants, -1) for d in directions]
    near_object, near_goal = [direction_to_angles(near_directions(d, generator, variants)) for d in directions]
    gaze = torch.stack((random, object_angles, goal_angles, near_object, near_goal), dim=1)
    model = CameraModel(**camera_cfg.camera_calibration)
    _, valid = model.project(angles_to_direction(gaze))
    valid &= (gaze.abs() <= angle_limit).all(-1)
    # Resample random entries against the physical frustum, never clip aligned targets.
    for _ in range(100):
        bad = ~valid[:, 0]
        if not bad.any():
            break
        gaze[:, 0][bad] = torch.empty(int(bad.sum()), 2).uniform_(-angle_limit, angle_limit, generator=generator)
        _, valid[:, 0] = model.project(angles_to_direction(gaze[:, 0]))
    if not valid.all():
        raise ValueError("Reset gaze targets exceed the physical camera frustum or gaze limits")
    identity = dict(
        version=1,
        source_sha256=artifact["content_sha256"],
        calibration=camera_cfg.camera_calibration,
        position=camera_cfg.offset.pos,
        rotation=camera_cfg.offset.rot,
        rack_pose=rack_pose,
        angle_limit=angle_limit,
        seed=seed,
        modes=RESET_MODES,
        vision_action_units="absolute_tilt_pan_radians",
        near_degrees=[2, 5],
        goal_mode="numbered" if numbered else "nearest",
    )
    return dict(
        identity=identity,
        identity_sha256=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        states=states,
        vision_action=gaze,
        object_direction=directions[0],
        goal_direction=directions[1],
    )


class CachedGazeAction(GazeAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.reset_mode = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.reset_variant = torch.zeros_like(self.reset_mode)
        self.source_rotation = self._target.new_tensor(self.camera.cfg.offset.rot)
        self.cache = None
        if env.cfg.mixed_gaze_resets and getattr(env.cfg, "retinal_geometry", "spherical") != "planar":
            params = env.event_manager.get_term_cfg("reset_from_dataset").params
            rack = env.cfg.scene.rack.init_state
            cache = build_reset_cache(
                params["dataset_path"],
                self.camera.cfg,
                (*rack.pos, *rack.rot),
                cfg.angle_limit,
                numbered=getattr(env.cfg, "goal_mode", "nearest") == "numbered",
            )
            output = Path(env.cfg.log_dir or Path.home() / ".cache/isaaclab_fovi/so101")
            output.mkdir(parents=True, exist_ok=True)
            torch.save(cache, output / "gaze_reset_cache.pt")
            self.cache = cache["vision_action"].to(self.device)

    def oracle(self, position):
        state = getattr(self._env, "so101_camera_randomization", None)
        direction = camera_directions(
            position, self.camera.get_world_positions(), self.source_rotation if state is None else state.rotation
        )
        if state is not None:
            visible = state.visibility(direction)
            direction = angles_to_direction(state.gaze_direction(direction))
            mount = torch.stack((direction[:, 0], direction[:, 2], -direction[:, 1]), -1)
            return fixation_action_target(
                mount, self._target, max_delta=self.cfg.max_delta, angle_limit=self.cfg.angle_limit, visible=visible
            )
        return calibrated_oracle(
            direction,
            self._target,
            self.calibration,
            max_delta=self.cfg.max_delta,
            angle_limit=self.cfg.angle_limit,
        )

    def fixation_error(self, position):
        state = getattr(self._env, "so101_camera_randomization", None)
        direction = camera_directions(
            position, self.camera.get_world_positions(), self.source_rotation if state is None else state.rotation
        )
        if state is not None:
            direction = angles_to_direction(state.gaze_direction(direction))
        actual = angles_to_direction(self._target)
        # atan2 remains accurate near perfect alignment, unlike acos(dot).
        return torch.atan2(
            torch.cross(actual, direction, dim=-1).norm(dim=-1),
            (actual * direction).sum(-1),
        )

    def reset(self, env_ids=None):
        super().reset(env_ids)
        state = getattr(self._env, "so101_camera_randomization", None)
        if state is not None and self._env.cfg.mixed_gaze_resets:
            self._reset_randomized_camera(env_ids, state)
            return
        if self.cache is None or not hasattr(self._env, "_so101_reset_row"):
            return
        ids = slice(None) if env_ids is None else env_ids
        rows = self._env._so101_reset_row[ids]
        modes = torch.randint(5, rows.shape, device=self.device)
        variants = torch.randint(self.cache.shape[2], rows.shape, device=self.device)
        self._target[ids] = self.cache[rows, modes, variants]
        self.reset_mode[ids], self.reset_variant[ids] = modes, variants
        # The cache contains absolute gaze state. The last incremental action and
        # velocity stay zero: reset is not a policy-generated camera movement.
        self.camera.set_calibrated_angles(self._target)

    def _reset_randomized_camera(self, env_ids, state):
        """Retain the cached mixture, deriving angles from the actual reset geometry."""
        from .goals import CAP_LABEL_POSITION, RACK_LABEL_OFFSETS

        env = self._env
        ids = torch.arange(env.num_envs, device=self.device)[slice(None) if env_ids is None else env_ids]
        if not len(ids) or not hasattr(env, "_so101_reset_vial_pose"):
            return
        vial = env._so101_reset_vial_pose[ids].clone()
        rack = env._so101_reset_rack_pose[ids].clone()
        vial[:, :3] += env.scene.env_origins[ids]
        rack[:, :3] += env.scene.env_origins[ids]
        obj, goal = target_centers(vial, rack, env.numbered_target_hole[ids], CAP_LABEL_POSITION, RACK_LABEL_OFFSETS)
        position = state.position[ids] + env.scene.env_origins[ids]
        modes = torch.randint(5, (len(ids),), device=self.device)
        target = torch.where(((modes == 2) | (modes == 4))[:, None], goal, obj)
        direction = camera_directions(target, position, state.rotation[ids])
        reference = direction.new_tensor((0, 1, 0)).expand_as(direction)
        tangent = torch.nn.functional.normalize(torch.cross(direction, reference, dim=-1), dim=-1)
        bitangent = torch.cross(direction, tangent, dim=-1)
        theta = torch.empty(len(ids), 1, device=self.device).uniform_(math.radians(2), math.radians(5))
        phi = torch.rand_like(theta) * (2 * math.pi)
        nearby = theta.cos() * direction + theta.sin() * (phi.cos() * tangent + phi.sin() * bitangent)
        direction = torch.where((modes >= 3)[:, None], nearby, direction)
        aligned = state.gaze_direction(direction, ids)
        random = torch.empty_like(aligned).uniform_(-self.cfg.angle_limit, self.cfg.angle_limit)
        from .camera_geometry import lens_visibility

        visible = lens_visibility(direction[:, None], state.intrinsics[ids], state.radial[ids], (640, 640)).squeeze(-1)
        valid = visible & (aligned.abs() <= self.cfg.angle_limit).all(-1)
        modes = torch.where(valid, modes, 0)
        # Random starts are bounded by the real source sensor too; a neutral
        # fallback prevents out-of-frame initialization at perturbed lens edges.
        _, random_valid = self.calibration.project(angles_to_direction(random))
        random = torch.where(random_valid[:, None], random, 0)
        self._target[ids] = torch.where((modes == 0)[:, None], random, aligned)
        self.reset_mode[ids], self.reset_variant[ids] = modes, 0
        self.camera.set_calibrated_angles(self._target)


@configclass
class CachedGazeActionCfg(GazeActionCfg):
    class_type: type = CachedGazeAction
