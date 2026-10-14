"""Event terms: dataset resets, and the joint, encoder and camera randomization Isaac Lab does not provide."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_from_euler_xyz, quat_mul, subtract_frame_transforms

from isaaclab_tutorial.tasks.place_vial.reset.dataset import load_reset_dataset

from .geometry import tabletop_vial_overlaps_rack

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ids(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    """Event managers pass ``None`` or ``slice(None)`` for all environments."""
    if env_ids is None:
        env_ids = slice(None)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)[env_ids]
    return env_ids


def _yaw_quat(angle: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(angle)
    return quat_from_euler_xyz(zeros, zeros, angle)


def phase_balanced_row_weights(phase: torch.Tensor, phase_weights: Sequence[float]) -> torch.Tensor:
    """Spread each phase's sampling weight uniformly over that phase's dataset rows."""
    weights = torch.as_tensor(phase_weights, device=phase.device, dtype=torch.float32)
    counts = torch.bincount(phase, minlength=len(weights))
    if len(counts) != len(weights):
        raise ValueError(f"phase_weights must contain exactly {len(counts)} values")
    if ((weights > 0) & (counts == 0)).any():
        raise ValueError("phase_weights request a phase with no dataset rows")
    return weights[phase] / counts.clamp_min(1)[phase]


class ResetFromDataset(ManagerTermBase):
    """Reset robot, vial and rack to physics-validated dataset rows, sampled by phase or in order.

    Home rows (phase 0) can be perturbed: vial position and heading, rack position and yaw, and the mat height. A
    vial that would overlap the rack is redrawn, falling back to its dataset pose. Mid-task rows are not perturbed.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.states = load_reset_dataset(cfg.params["dataset_path"], device=env.device)["states"]
        self.row_weights = phase_balanced_row_weights(self.states["phase"], cfg.params["phase_weights"])
        self.sequential_rows = self.row_weights.nonzero().flatten()
        self.cursor = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        dataset_path: str,
        phase_weights: Sequence[float],
        sequential: bool = False,
        home_position_noise: float = 0.0,
        home_heading_noise: float = 0.0,
        home_rack_clearance: float = 0.0,
        home_rack_position_noise: float = 0.0,
        home_rack_yaw_noise: float = 0.0,
        support_height_range: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        ids = _ids(env, env_ids)
        n = len(ids)
        if sequential:
            rows = self.sequential_rows[(torch.arange(n, device=env.device) + self.cursor) % len(self.sequential_rows)]
            self.cursor = (self.cursor + n) % len(self.sequential_rows)
        else:
            rows = torch.multinomial(self.row_weights, n, replacement=True)
        states = {name: value[rows] for name, value in self.states.items()}
        home = (states["phase"] == 0)[:, None]

        robot = env.scene["robot"]
        zeros = torch.zeros_like(states["joint_position"])
        robot.write_joint_position_to_sim_index(position=states["joint_position"], env_ids=ids)
        robot.write_joint_velocity_to_sim_index(velocity=zeros, env_ids=ids)
        robot.set_joint_position_target_index(target=states["joint_target"], env_ids=ids)
        robot.set_joint_velocity_target_index(target=zeros, env_ids=ids)

        def uniform(shape, bound):
            return torch.empty(shape, device=env.device).uniform_(-bound, bound) * home

        rack_pose = env.scene["rack"].data.default_root_pose.torch[ids].clone()
        rack_pose[:, :2] += uniform((n, 2), home_rack_position_noise)
        rack_pose[:, 3:] = quat_mul(_yaw_quat(uniform((n, 1), home_rack_yaw_noise)[:, 0]), rack_pose[:, 3:])
        vial_pose = states["vial_pose"].clone()
        vial_pose[:, 3:] = quat_mul(_yaw_quat(uniform((n, 1), home_heading_noise)[:, 0]), vial_pose[:, 3:])
        vial_pose[:, :2] += uniform((n, 2), home_position_noise)
        for _ in range(16):
            redraw = tabletop_vial_overlaps_rack(vial_pose, rack_pose, home_rack_clearance) & home[:, 0]
            if not redraw.any():
                break
            vial_pose[redraw, :2] = states["vial_pose"][redraw, :2] + uniform((n, 2), home_position_noise)[redraw]
        else:
            vial_pose[redraw, :2] = states["vial_pose"][redraw, :2]
            rack_pose[redraw] = env.scene["rack"].data.default_root_pose.torch[ids[redraw]]

        if "support" in env.scene.keys():  # noqa: SIM118 (InteractiveScene has no __contains__)
            # The 5 mm kinematic mat overlaps the desk; its exposed height spans bare desk to full mat.
            height = torch.empty(n, device=env.device).uniform_(*support_height_range) * home[:, 0]
            if not hasattr(env, "_so101_support_height"):
                env._so101_support_height = torch.zeros(env.num_envs, device=env.device)
            env._so101_support_height[ids] = height
            support = env.scene["support"]
            support_pose = support.data.default_root_pose.torch[ids].clone()
            support_pose[:, 2] += height
            support_pose[:, :3] += env.scene.env_origins[ids]
            support.write_root_pose_to_sim_index(root_pose=support_pose, env_ids=ids)
            vial_pose[:, 2] += height
            rack_pose[:, 2] += height

        # Read by the success term (restored milestones) and by evaluation (initial conditions).
        for name, value in (
            ("row", rows),
            ("grasped", states["grasped"]),
            ("lifted", states["lifted"]),
            ("vial_pose", vial_pose),
            ("rack_pose", rack_pose),
        ):
            attribute = f"_so101_reset_{name}"
            if not hasattr(env, attribute):
                setattr(env, attribute, value.new_zeros((env.num_envs, *value.shape[1:])))
            getattr(env, attribute)[ids] = value
        for name, pose in (("vial", vial_pose), ("rack", rack_pose)):
            pose = pose.clone()
            pose[:, :3] += env.scene.env_origins[ids]
            env.scene[name].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            env.scene[name].write_root_velocity_to_sim_index(
                root_velocity=torch.zeros(n, 6, device=env.device), env_ids=ids
            )


class randomize_joint_viscous_friction(ManagerTermBase):
    """Scale passive viscous joint friction per environment, relative to the startup values.

    Isaac Lab randomizes Coulomb friction and armature (``randomize_joint_parameters``) and drive gains
    (``randomize_actuator_gains``) but not viscous friction.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self.default_viscous = self.asset.data.joint_viscous_friction_coeff.torch.clone()

    def __call__(
        self, env: ManagerBasedRLEnv, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg, scale_range: tuple
    ) -> None:
        ids = _ids(env, env_ids)
        joint_ids = asset_cfg.joint_ids
        viscous = self.default_viscous[ids][:, joint_ids]
        self.asset.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=self.asset.data.joint_friction_coeff.torch[ids][:, joint_ids],
            joint_viscous_friction_coeff=viscous * torch.empty_like(viscous).uniform_(*scale_range),
            joint_ids=None if joint_ids == slice(None) else joint_ids,
            env_ids=ids,
        )


def randomize_encoder_bias(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None, bound: float = 0.01) -> None:
    """Sample an episode-constant joint calibration error [rad], read by the encoder observations."""
    ids = _ids(env, env_ids)
    if not hasattr(env, "_so101_encoder_bias"):
        env._so101_encoder_bias = torch.zeros(env.num_envs, 6, device=env.device)
    env._so101_encoder_bias[ids] = torch.empty(len(ids), 6, device=env.device).uniform_(-bound, bound)


class randomize_wrist_camera_mount(ManagerTermBase):
    """Offset the wrist camera from its nominal mount by a per-episode translation [m] and rotation [rad]."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.nominal_position = self.nominal_orientation = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        position_range: float,
        rotation_range: float,
    ) -> None:
        ids = _ids(env, env_ids)
        camera = env.scene.sensors[sensor_cfg.name]
        robot = env.scene[asset_cfg.name]
        # Dataset resets just wrote joint positions; refresh kinematics before reading the carrier link.
        env.sim.forward()
        carrier_position = robot.data.body_pos_w.torch[:, asset_cfg.body_ids[0]]
        carrier_orientation = robot.data.body_quat_w.torch[:, asset_cfg.body_ids[0]]
        if self.nominal_position is None:
            self.nominal_position, self.nominal_orientation = subtract_frame_transforms(
                carrier_position, carrier_orientation, camera.data.pos_w.torch, camera.data.quat_w_ros.torch
            )
        translation = torch.empty(len(ids), 3, device=env.device).uniform_(-position_range, position_range)
        rotation = quat_from_euler_xyz(*torch.empty_like(translation).uniform_(-rotation_range, rotation_range).T)
        local_position, local_orientation = combine_frame_transforms(
            self.nominal_position[ids], self.nominal_orientation[ids], translation, rotation
        )
        position, orientation = combine_frame_transforms(
            carrier_position[ids], carrier_orientation[ids], local_position, local_orientation
        )
        camera.set_world_poses(position, orientation, env_ids=ids, convention="ros")
        # Reading camera.data above may have rendered the old pose; invalidate it for the new episode.
        camera.reset(env_ids=ids)
