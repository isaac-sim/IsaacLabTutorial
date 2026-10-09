"""Reset events for validated task-horizon state replay."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import EventTermCfg, ManagerTermBase

from isaaclab_tutorial.tasks.place_vial.mdp.geometry import tabletop_vial_overlaps_rack
from isaaclab_tutorial.tasks.place_vial.reset.dataset import load_reset_dataset

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_INTEGER_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


def _phase_balanced_row_weights(phase: torch.Tensor, phase_weights: Sequence[float]) -> torch.Tensor:
    """Spread each requested phase probability uniformly over that phase's rows."""
    if phase.ndim != 1 or phase.numel() == 0:
        raise ValueError("phase must be a nonempty one-dimensional tensor")
    if phase.dtype not in _INTEGER_DTYPES or bool((phase < 0).any()):
        raise ValueError("phase must contain nonnegative integers")

    phase_count = int(phase.max().item()) + 1
    weights = torch.as_tensor(phase_weights, device=phase.device, dtype=torch.float32)
    if weights.ndim != 1 or len(weights) != phase_count:
        raise ValueError(f"phase_weights must contain exactly {phase_count} values")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()) or not bool(weights.any()):
        raise ValueError("phase_weights must be finite, nonnegative, and not all zero")

    eligible = weights[phase] > 0.0
    eligible_counts = torch.bincount(phase[eligible], minlength=phase_count)
    missing = (weights > 0.0) & (eligible_counts == 0)
    if bool(missing.any()):
        missing_phases = missing.nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"Reset curriculum has no eligible rows for phases {missing_phases}")
    per_phase_count = eligible_counts.clamp_min(1).to(weights.dtype)
    row_weights = weights[phase] / per_phase_count[phase]
    return torch.where(eligible, row_weights, torch.zeros_like(row_weights))


def _ids(env: ManagerBasedRLEnv, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor:
    """Normalize event-manager environment indices."""
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    raw_ids = torch.as_tensor(env_ids)
    if raw_ids.ndim > 1 or raw_ids.dtype not in _INTEGER_DTYPES:
        raise ValueError("env_ids must contain integers in a scalar or one-dimensional sequence")
    if raw_ids.device.type == "cpu":
        if bool(((raw_ids < 0) | (raw_ids >= env.num_envs)).any()):
            raise ValueError(f"env_ids must lie in [0, {env.num_envs - 1}]")
        if raw_ids.numel() != torch.unique(raw_ids).numel():
            raise ValueError("env_ids must not contain duplicates")
    return raw_ids.to(device=env.device, dtype=torch.long).reshape(-1)


def _reset_progress_seed(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    *,
    phase: torch.Tensor,
    grasped: torch.Tensor,
    lifted: torch.Tensor,
) -> None:
    """Publish reset-row history for the instance-owned success term."""
    if not hasattr(env, "_so101_reset_phase"):
        env._so101_reset_phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._so101_reset_grasped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._so101_reset_lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._so101_reset_phase[env_ids] = phase
    env._so101_reset_grasped[env_ids] = grasped
    env._so101_reset_lifted[env_ids] = lifted


def _reset_controller_seed(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    joint_target: torch.Tensor | None,
) -> None:
    """Publish the target consumed when the action manager resets after events."""
    if not hasattr(env, "_so101_reset_joint_target"):
        env._so101_reset_joint_target = torch.zeros((env.num_envs, 6), device=env.device)
        env._so101_use_reset_joint_target = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if joint_target is None:
        env._so101_use_reset_joint_target[env_ids] = False
    else:
        env._so101_reset_joint_target[env_ids] = joint_target
        env._so101_use_reset_joint_target[env_ids] = True


class ResetFromDataset(ManagerTermBase):
    """Replay physics-validated reset rows, sampled by phase weight or in deterministic order."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        artifact = load_reset_dataset(cfg.params["dataset_path"], device=env.device)
        self.states = artifact["states"]
        self.row_count = int(artifact["row_count"])
        self._cursor = 0
        self._noise_generator = torch.Generator(device=env.device).manual_seed(env.cfg.seed or 0)
        phase_weights = cfg.params.get("phase_weights")
        self.row_weights = None
        if phase_weights is not None:
            self.row_weights = _phase_balanced_row_weights(self.states["phase"], phase_weights)
        self.sequential_rows = (
            torch.arange(self.row_count, device=env.device)
            if self.row_weights is None
            else self.row_weights.nonzero(as_tuple=False).flatten()
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int] | torch.Tensor | slice,
        dataset_path: str,
        sequential: bool = False,
        phase_weights: tuple[float, ...] | None = None,
        home_position_noise: float = 0.0,
        home_rack_clearance: float = 0.0,
        home_rack_position_noise: float = 0.0,
        home_rack_yaw_noise: float = 0.0,
        home_heading_noise: float = 0.0,
        support_height_range: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Write reset states, optionally perturbing home-start vial XY by ``home_position_noise`` [m]."""
        del dataset_path, phase_weights
        ids = _ids(env, env_ids)
        if ids.numel() == 0:
            return
        if sequential:
            indices = (torch.arange(ids.numel(), device=env.device) + self._cursor).remainder(
                self.sequential_rows.numel()
            )
            rows = self.sequential_rows[indices]
            self._cursor = (self._cursor + ids.numel()) % self.sequential_rows.numel()
        elif self.row_weights is None:
            rows = torch.randint(self.row_count, (ids.numel(),), device=env.device)
        else:
            rows = torch.multinomial(self.row_weights, ids.numel(), replacement=True)

        if not hasattr(env, "_so101_reset_row"):
            env._so101_reset_row = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._so101_reset_row[ids] = rows

        robot = env.scene["robot"]
        joint_position = self.states["joint_position"][rows]
        joint_target = self.states["joint_target"][rows]
        joint_velocity = torch.zeros_like(joint_position)
        robot.write_joint_position_to_sim_index(position=joint_position, env_ids=ids)
        robot.write_joint_velocity_to_sim_index(velocity=joint_velocity, env_ids=ids)
        robot.set_joint_position_target_index(target=joint_target, env_ids=ids)
        robot.set_joint_velocity_target_index(target=joint_velocity, env_ids=ids)
        # Synchronize the jaw latch because the articulation cache can still
        # contain the terminal sample during a partial reset.
        env.action_manager.get_term("gripper_action").seed_joint_target(ids, joint_target[:, -1:])
        _reset_controller_seed(env, ids, joint_target)

        vial_pose = self.states["vial_pose"][rows].clone()
        rack_pose = env.scene["rack"].data.default_root_pose.torch[ids].clone()
        home = self.states["phase"][rows] == 0
        if min(home_rack_position_noise, home_rack_yaw_noise, home_heading_noise, *support_height_range) < 0:
            raise ValueError("Placement randomization bounds must be nonnegative")
        if support_height_range[1] < support_height_range[0]:
            raise ValueError("Support height range must be ordered")
        if home_rack_position_noise or home_rack_yaw_noise or home_heading_noise:
            from isaaclab.utils.math import quat_mul

            rack_pose[:, :2] += (
                torch.empty_like(rack_pose[:, :2]).uniform_(
                    -home_rack_position_noise, home_rack_position_noise, generator=self._noise_generator
                )
                * home[:, None]
            )
            for pose, bound in ((rack_pose, home_rack_yaw_noise), (vial_pose, home_heading_noise)):
                angle = (
                    torch.empty(len(ids), device=env.device).uniform_(-bound, bound, generator=self._noise_generator)
                    * home
                )
                rotation = torch.zeros((len(ids), 4), device=env.device)
                rotation[:, 2] = torch.sin(angle / 2)
                rotation[:, 3] = torch.cos(angle / 2)
                pose[:, 3:] = quat_mul(rotation, pose[:, 3:])
        if home_position_noise < 0 or home_rack_clearance < 0:
            raise ValueError("Home position noise and rack clearance must be nonnegative [m]")
        if home_position_noise or home_rack_position_noise or home_rack_yaw_noise or home_heading_noise:
            noise = torch.empty((ids.numel(), 2), device=env.device).uniform_(
                -home_position_noise, home_position_noise, generator=self._noise_generator
            )
            vial_pose[:, :2] += noise * home[:, None]
            # Jitter must not teleport a validated tabletop pose into the rack's solid lower deck.
            for _ in range(16):
                rejected = home & tabletop_vial_overlaps_rack(vial_pose, rack_pose, home_rack_clearance)
                retry = rejected.nonzero(as_tuple=False).flatten()
                if retry.numel() == 0:
                    break
                offset = torch.empty((len(retry), 2), device=env.device).uniform_(
                    -home_position_noise, home_position_noise, generator=self._noise_generator
                )
                vial_pose[retry, :2] = self.states["vial_pose"][rows[retry], :2] + offset
            # A physical reset row is not necessarily clear under the conservative
            # footprint test. Try bounded corners before declaring an impossible draw.
            rejected = home & tabletop_vial_overlaps_rack(vial_pose, rack_pose, home_rack_clearance)
            fallback = rejected.clone()
            rack_pose[fallback] = env.scene["rack"].data.default_root_pose.torch[ids[fallback]]
            for x, y in ((0, 0), (0, -1), (-1, -1), (1, -1), (-1, 0), (1, 0), (0, 1), (-1, 1), (1, 1)):
                if not bool(rejected.any()):
                    break
                vial_pose[rejected] = self.states["vial_pose"][rows[rejected]]
                vial_pose[rejected, :2] += vial_pose.new_tensor((x, y)) * home_position_noise
                rejected = fallback & tabletop_vial_overlaps_rack(vial_pose, rack_pose, home_rack_clearance)
            if bool(rejected.any()):
                raise ValueError("No collision-free home reset within configured position bounds")
        if "support" in env.scene.keys():  # noqa: SIM118 (InteractiveScene has no __contains__)
            # A 5 mm kinematic pad overlaps the desk; its exposed height spans bare desk to mat.
            height = (
                torch.empty(len(ids), device=env.device).uniform_(
                    *support_height_range, generator=self._noise_generator
                )
                * home
            )
            if not hasattr(env, "_so101_support_height"):
                env._so101_support_height = torch.zeros(env.num_envs, device=env.device)
            env._so101_support_height[ids] = height
            support = env.scene["support"]
            support_pose = support.data.default_root_pose.torch[ids].clone()
            support_pose[:, 2] += height
            support_pose[:, :3] += env.scene.env_origins[ids]
            support.write_root_pose_to_sim_index(root_pose=support_pose, env_ids=ids)
            support.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((len(ids), 6), device=env.device), env_ids=ids
            )
            vial_pose[:, 2] += height
            rack_pose[:, 2] += height
        if not hasattr(env, "_so101_reset_vial_pose"):
            env._so101_reset_vial_pose = torch.zeros((env.num_envs, 7), device=env.device)
        env._so101_reset_vial_pose[ids] = vial_pose
        vial_pose[:, :3] += env.scene.env_origins[ids]
        vial = env.scene["vial"]
        vial.write_root_pose_to_sim_index(root_pose=vial_pose, env_ids=ids)
        vial.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((ids.numel(), 6), device=env.device),
            env_ids=ids,
        )

        if not hasattr(env, "_so101_reset_rack_pose"):
            env._so101_reset_rack_pose = torch.zeros((env.num_envs, 7), device=env.device)
        env._so101_reset_rack_pose[ids] = rack_pose
        rack = env.scene["rack"]
        rack_pose[:, :3] += env.scene.env_origins[ids]
        rack.write_root_pose_to_sim_index(root_pose=rack_pose, env_ids=ids)
        rack.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((ids.numel(), 6), device=env.device),
            env_ids=ids,
        )
        _reset_progress_seed(
            env,
            ids,
            phase=self.states["phase"][rows],
            grasped=self.states["grasped"][rows],
            lifted=self.states["lifted"][rows],
        )


def clear_reset_progress(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Mark ordinary tabletop resets as the beginning of the task."""
    ids = _ids(env, env_ids)
    _reset_progress_seed(
        env,
        ids,
        phase=torch.zeros_like(ids),
        grasped=torch.zeros_like(ids, dtype=torch.bool),
        lifted=torch.zeros_like(ids, dtype=torch.bool),
    )
    _reset_controller_seed(env, ids, None)


class RandomizeJointViscousFriction(ManagerTermBase):
    """Scale the passive viscous joint friction of an articulation per environment.

    Isaac Lab randomizes Coulomb friction and armature (:func:`randomize_joint_parameters`) and the drive gains
    (:func:`randomize_actuator_gains`) but not the passive viscous term, which is the joint quantity that differs
    most between the Newton and OV PhysX backends for the SO-101. The scale is sampled uniformly per environment and
    joint from ``scale_range`` and applied to the values the asset had at startup.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params["asset_cfg"]
        self.asset = env.scene[self.asset_cfg.name]
        viscous = self.asset.data.joint_viscous_friction_coeff
        self.default_viscous = (viscous.torch if hasattr(viscous, "torch") else viscous).clone()
        low, high = cfg.params["scale_range"]
        if not 0.0 < low <= high:
            raise ValueError(f"scale_range must satisfy 0 < low <= high, got {cfg.params['scale_range']}")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        asset_cfg,
        scale_range: tuple[float, float],
    ) -> None:
        ids = _ids(env, env_ids)
        if ids.numel() == 0:
            return
        joint_ids = self.asset_cfg.joint_ids
        base = self.default_viscous[ids]
        if joint_ids != slice(None):
            base = base[:, joint_ids]
        scale = torch.empty_like(base).uniform_(scale_range[0], scale_range[1])
        friction = self.asset.data.joint_friction_coeff
        friction = (friction.torch if hasattr(friction, "torch") else friction)[ids]
        if joint_ids != slice(None):
            friction = friction[:, joint_ids]
        self.asset.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=friction,
            joint_viscous_friction_coeff=base * scale,
            joint_ids=None if joint_ids == slice(None) else joint_ids,
            env_ids=ids,
        )


def randomize_encoder_bias(env, env_ids, bound: float = 0.01):
    """Sample small persistent coordinate errors shared by position and target feedback."""
    if bound < 0:
        raise ValueError("Encoder bias bound must be nonnegative")
    ids = _ids(env, env_ids)
    if not hasattr(env, "_so101_encoder_bias"):
        env._so101_encoder_bias = torch.zeros((env.num_envs, 6), device=env.device)
    env._so101_encoder_bias[ids] = torch.empty((len(ids), 6), device=env.device).uniform_(-bound, bound)


def configure_support_rolling_contacts(builder, rolling_range, torsional_range):
    """Enable rolling resistance only at the table/support, preserving jaw/rack contact dimensions.

    Pair friction uses the maximum of both shapes. Clear previously inactive
    rolling coefficients on the other shapes so they cannot override the small
    support coefficients when the support requests a six-dimensional contact.
    """
    for bounds in (rolling_range, torsional_range):
        if not 0 <= bounds[0] <= bounds[1]:
            raise ValueError("Rolling/torsional ranges must be nonnegative and ordered")
    per_world = {}
    condim = builder.custom_attributes["mujoco:condim"].values
    for index, label in enumerate(builder.shape_label):
        world = builder.shape_world[index]
        if world not in per_world:
            u, v = torch.rand(2).tolist()
            per_world[world] = (
                rolling_range[0] + u * (rolling_range[1] - rolling_range[0]),
                torsional_range[0] + v * (torsional_range[1] - torsional_range[0]),
            )
        surface = "/Support/" in label or "/Desk/" in label
        builder.shape_material_mu_rolling[index] = per_world[world][0] if surface else 0.0
        builder.shape_material_mu_torsional[index] = per_world[world][1] if surface else 0.0
        if surface:
            condim[index] = 6
