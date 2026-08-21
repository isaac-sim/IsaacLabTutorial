"""Observations, rewards, history, and terminations for vial placement."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, subtract_frame_transforms

from .geometry import inside_bounds, rack_local_position, vertical_alignment
from .progress import PlacementProgress

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import Camera, ContactSensor


RACK_LOWER = (-0.035, -0.055, 0.010)
RACK_UPPER = (0.035, 0.055, 0.100)
FIXED_FINGERTIP_OFFSET = (-0.01289, -0.00022, -0.0986)
MOVING_FINGERTIP_OFFSET = (-0.0068, -0.0766, 0.0189)
# The authored spawn root is 0.025 m, but the horizontal vial settles with its
# root at about 0.012 m on the mat under Newton. Use the simulated rest pose so
# height shaping has no dead zone while still being zero on the tabletop.
VIAL_REST_HEIGHT = 0.012
FIXED_FINGERTIP_REST_HEIGHT = 0.006
LIFT_HEIGHT = 0.05
GRASP_LIFT_HEIGHT = 0.14
INSERTION_JOINT_GOAL = (-1.211648, 0.187987, 0.401624, 0.366479, 0.786474)


def _tensor(value):
    """Return the torch view of an Isaac Lab proxy array."""
    return value.torch if hasattr(value, "torch") else value


def _finite_reward(value: torch.Tensor) -> torch.Tensor:
    """Map invalid terminal-state shaping values to a neutral reward."""
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def _contact(env: ManagerBasedRLEnv, name: str, threshold: float = 0.25) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[name]
    # ``net_forces_w`` includes contacts with every body (notably the rack),
    # even when a filter is configured.  Grasp and release semantics must use
    # the vial-only partner matrix produced by ``filter_prim_paths_expr``.
    forces = _tensor(sensor.data.force_matrix_w)
    return torch.linalg.vector_norm(forces, dim=-1).reshape(env.num_envs, -1).amax(dim=-1) > threshold


def bilateral_contact(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Whether the vial is touching both jaws."""
    return _contact(env, "fixed_jaw_contact") & _contact(env, "moving_jaw_contact")


def retained_grasp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Whether physical jaw contact or the post-contact constraint retains the vial."""
    action_term = env.action_manager.get_term("joint_delta")
    grasp_latched = getattr(action_term, "_grasp_latched", None)
    if grasp_latched is None:
        return bilateral_contact(env)
    return bilateral_contact(env) | grasp_latched


def fingertip_positions_w(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Return both simplified contact-pad centers in world coordinates."""
    robot: Articulation = env.scene["robot"]
    fixed_id = robot.find_bodies("gripper", preserve_order=True)[0]
    moving_id = robot.find_bodies("moving_jaw_so101_v1", preserve_order=True)[0]
    fixed_pos = _tensor(robot.data.body_pos_w)[:, fixed_id].squeeze(1)
    fixed_quat = _tensor(robot.data.body_quat_w)[:, fixed_id].squeeze(1)
    moving_pos = _tensor(robot.data.body_pos_w)[:, moving_id].squeeze(1)
    moving_quat = _tensor(robot.data.body_quat_w)[:, moving_id].squeeze(1)
    fixed = fixed_pos + quat_apply(fixed_quat, fixed_pos.new_tensor(FIXED_FINGERTIP_OFFSET).expand_as(fixed_pos))
    moving = moving_pos + quat_apply(moving_quat, moving_pos.new_tensor(MOVING_FINGERTIP_OFFSET).expand_as(moving_pos))
    return fixed, moving


def _grasp_center_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the midpoint between the two fingertip contact pads."""
    fixed, moving = fingertip_positions_w(env)
    return 0.5 * (fixed + moving)


def _placement_values(env: ManagerBasedRLEnv):
    vial: RigidObject = env.scene["vial"]
    rack: RigidObject = env.scene["rack"]
    vial_pos = _tensor(vial.data.root_pos_w)
    vial_quat = _tensor(vial.data.root_quat_w)
    rack_pos = _tensor(rack.data.root_pos_w)
    rack_quat = _tensor(rack.data.root_quat_w)
    local = rack_local_position(vial_pos, rack_pos, rack_quat)
    local = torch.nan_to_num(local, nan=1.0e3, posinf=1.0e3, neginf=-1.0e3)
    alignment = _finite_reward(vertical_alignment(vial_quat))
    speed = torch.nan_to_num(
        torch.linalg.vector_norm(_tensor(vial.data.root_lin_vel_w), dim=-1),
        nan=1.0e3,
        posinf=1.0e3,
        neginf=1.0e3,
    )
    no_contact = ~(_contact(env, "fixed_jaw_contact") | _contact(env, "moving_jaw_contact"))
    action_term = env.action_manager.get_term("joint_delta")
    grasp_latched = getattr(action_term, "_grasp_latched", torch.zeros_like(no_contact))
    released = no_contact & ~grasp_latched
    placed = inside_bounds(local, RACK_LOWER, RACK_UPPER)
    return local, alignment, speed, released, placed


class PlacementHistoryTerm(ManagerTermBase):
    """Instance-owned success term; manager resets preserve unrelated environments."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.progress = _history(env)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.progress.reset(env_ids)
        stages = getattr(self._env, "_so101_assisted_stage", None)
        if stages is not None:
            ids = (
                torch.arange(self.num_envs, device=self.device)
                if env_ids is None
                else torch.as_tensor(env_ids, device=self.device)
            )
            # Every assisted pose represents a vial that has already been
            # grasped and lifted.  Seed those irreversible prerequisites
            # explicitly instead of relying on a reset pose happening to lie
            # above the absolute lift-height threshold.  Canonical resets use
            # stage -1 and therefore receive no history.
            assisted_start = ids[stages[ids] >= 0]
            self.progress.grasped[assisted_start] = True
            self.progress.lifted[assisted_start] = True
            # Stage 8 is authored inside the rack with a retained grasp.  It is
            # the release curriculum state, so arm masking must be active from
            # its first policy action rather than after two exploratory steps.
            release_start = ids[stages[ids] == 8]
            self.progress.release_ready[release_start] = True

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        vial: RigidObject = env.scene["vial"]
        vial_z = _tensor(vial.data.root_pos_w)[:, 2]
        local, alignment, speed, released, placed = _placement_values(env)
        valid = placed & (alignment > 0.8) & released & (speed < 0.1)
        # The vial may already be inside the broad rack bounds while still
        # hovering high enough to tip on release.  Require the policy to lower
        # it to within 12 mm of the authored rest center before opening.
        insertion_ready = local[:, 2] < 0.0475
        valid_grasped = placed & insertion_ready & (alignment > 0.8) & retained_grasp(env)
        success = self.progress.update(
            bilateral_contact(env),
            vial_z > 0.065,
            valid,
            env.episode_length_buf,
            valid_grasped,
        )
        log = env.extras.setdefault("log", {})
        log["Metrics/grasp_rate"] = self.progress.grasped.float().mean()
        log["Metrics/lift_rate"] = self.progress.lifted.float().mean()
        log["Metrics/vial_height_m"] = vial_z.mean()
        fixed_fingertip, moving_fingertip = fingertip_positions_w(env)
        log["Metrics/fixed_fingertip_height_m"] = fixed_fingertip[:, 2].mean()
        log["Metrics/moving_fingertip_height_m"] = moving_fingertip[:, 2].mean()
        robot: Articulation = env.scene["robot"]
        joint_pos = _tensor(robot.data.joint_pos)
        for joint_id, joint_name in enumerate(robot.joint_names):
            log[f"Diagnostics/joint_pos/{joint_name}"] = joint_pos[:, joint_id].mean()
        log["Diagnostics/grasp_center_x_m"] = (0.5 * (fixed_fingertip[:, 0] + moving_fingertip[:, 0])).mean()
        log["Diagnostics/grasp_center_y_m"] = (0.5 * (fixed_fingertip[:, 1] + moving_fingertip[:, 1])).mean()
        log["Diagnostics/vertical_alignment"] = alignment.mean()
        log["Diagnostics/rack_local_x_m"] = local[:, 0].mean()
        log["Diagnostics/rack_local_y_m"] = local[:, 1].mean()
        log["Diagnostics/rack_local_z_m"] = local[:, 2].mean()
        log["Metrics/placement_rate"] = valid.float().mean()
        log["Metrics/bilateral_contact_rate"] = bilateral_contact(env).float().mean()
        log["Metrics/rack_hold_rate"] = valid_grasped.float().mean()
        log["Metrics/release_ready_rate"] = self.progress.release_ready.float().mean()
        log["Metrics/success_rate"] = success.float().mean()
        action_term = env.action_manager.get_term("joint_delta")
        log["Diagnostics/grasp_latch_rate"] = action_term._grasp_latched.float().mean()
        pending_grasp = getattr(env, "_so101_pending_assisted_grasp", None)
        if pending_grasp is not None:
            log["Diagnostics/pending_assisted_grasp_rate"] = pending_grasp.float().mean()
        completed = self.progress.time_to_success >= 0
        if completed.any():
            log["Metrics/time_to_success_s"] = self.progress.time_to_success[completed].float().mean() * env.step_dt
        return success


def _history(env: ManagerBasedRLEnv) -> PlacementProgress:
    # Observations are constructed before the termination manager, so the shared
    # per-environment state must be available during observation shape inference.
    progress = getattr(env, "_so101_placement_progress", None)
    if progress is None:
        progress = PlacementProgress(env.num_envs, env.device, stable_steps=15)
        env._so101_placement_progress = progress
    return progress


def progress_flags(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return irreversible grasp/lift flags."""
    history = _history(env)
    return torch.stack((history.grasped, history.lifted), dim=-1).float()


def _robot_root_pose(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the fixed robot base pose used as the task reference frame."""
    robot: Articulation = env.scene["robot"]
    return _tensor(robot.data.root_link_pos_w), _tensor(robot.data.root_link_quat_w)


def body_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return a robot body state expressed in the robot base frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    ids = asset_cfg.body_ids
    root_pos_w, root_quat_w = _robot_root_pose(env)
    body_pos_w = _tensor(robot.data.body_pos_w)[:, ids].reshape(-1, 3)
    body_quat_w = _tensor(robot.data.body_quat_w)[:, ids].reshape(-1, 4)
    bodies_per_env = body_pos_w.shape[0] // env.num_envs
    root_pos_w = root_pos_w.repeat_interleave(bodies_per_env, dim=0)
    root_quat_w = root_quat_w.repeat_interleave(bodies_per_env, dim=0)
    body_pos_b, body_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
    body_lin_vel_b = quat_apply_inverse(root_quat_w, _tensor(robot.data.body_lin_vel_w)[:, ids].reshape(-1, 3))
    body_ang_vel_b = quat_apply_inverse(root_quat_w, _tensor(robot.data.body_ang_vel_w)[:, ids].reshape(-1, 3))
    return torch.cat(
        (body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b),
        dim=-1,
    ).reshape(env.num_envs, -1)


def rigid_object_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return a rigid-object state expressed in the robot base frame."""
    obj: RigidObject = env.scene[asset_cfg.name]
    root_pos_w, root_quat_w = _robot_root_pose(env)
    obj_pos_b, obj_quat_b = subtract_frame_transforms(
        root_pos_w,
        root_quat_w,
        _tensor(obj.data.root_pos_w),
        _tensor(obj.data.root_quat_w),
    )
    return torch.cat(
        (
            obj_pos_b,
            obj_quat_b,
            quat_apply_inverse(root_quat_w, _tensor(obj.data.root_lin_vel_w)),
            quat_apply_inverse(root_quat_w, _tensor(obj.data.root_ang_vel_w)),
        ),
        dim=-1,
    )


def rack_relative_target(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return vial position in the rack frame."""
    local, _, _, _, _ = _placement_values(env)
    return local


class NormalizedCameraImage(ManagerTermBase):
    """Read a fresh RGB image as normalized NCHW floats."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
        camera: Camera = env.scene.sensors[sensor_cfg.name]
        image = _tensor(camera.data.output["rgb"])[..., :3]
        return image.permute(0, 3, 1, 2).contiguous().float().div(255.0)


def reaching_reward(env: ManagerBasedRLEnv, std: float = 0.10) -> torch.Tensor:
    """Reward bringing the center of the jaws to the vial.

    The tanh kernel follows Isaac Lab's manipulation rewards: it keeps a
    useful gradient over the reachable workspace while approaching one near
    the object.  Using the jaw midpoint avoids rewarding the wrist origin.
    """
    vial: RigidObject = env.scene["vial"]
    distance = torch.linalg.vector_norm(_grasp_center_w(env) - _tensor(vial.data.root_pos_w), dim=-1)
    return _finite_reward(1.0 - torch.tanh(distance / std))


def bilateral_contact_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Maintain bilateral contact through transport, then permit release."""
    history = _history(env)
    return (retained_grasp(env) & ~history.release_ready).float()


def lift_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Give smooth lift progress only until the lift milestone is reached.

    Keeping this saturated reward active during transport lets the policy earn
    more return by hovering for the rest of the episode than by attempting the
    harder placement.  Later phases are shaped exclusively by upright,
    transport, insertion, and release terms.
    """
    vial: RigidObject = env.scene["vial"]
    height_progress = ((_tensor(vial.data.root_pos_w)[:, 2] - VIAL_REST_HEIGHT) / LIFT_HEIGHT).clamp(0.0, 1.0)
    history = _history(env)
    grasp_gate = history.grasped | retained_grasp(env)
    return _finite_reward((grasp_gate & ~history.lifted).float() * height_progress)


def grasp_lift_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward raising the hand after a grasp has been established.

    Contact sensors can flicker for a control step while a constrained object
    starts moving.  Gating this shaping term on live contact made that first
    upward motion locally worse than holding the vial against the mat.  The
    irreversible grasp milestone supplies the robust stage gate; the separate
    object-height reward still makes abandoning the vial unprofitable.
    """
    fixed_fingertip, _ = fingertip_positions_w(env)
    height_progress = ((fixed_fingertip[:, 2] - FIXED_FINGERTIP_REST_HEIGHT) / GRASP_LIFT_HEIGHT).clamp(0.0, 1.0)
    history = _history(env)
    active = (history.grasped | retained_grasp(env)) & ~history.lifted
    return _finite_reward(active.float() * height_progress)


def lifting_velocity_reward(env: ManagerBasedRLEnv, max_speed: float = 0.25) -> torch.Tensor:
    """Give signed vertical-velocity shaping while both fingertips retain contact.

    Keeping the downward half of the signal prevents a policy from earning
    return by oscillating the vial without making net height progress.
    """
    vial: RigidObject = env.scene["vial"]
    vertical_speed = (_tensor(vial.data.root_lin_vel_w)[:, 2] / max_speed).clamp(-1.0, 1.0)
    history = _history(env)
    return _finite_reward((history.grasped & ~history.lifted).float() * vertical_speed)


def upright_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    _, alignment, _, _, _ = _placement_values(env)
    history = _history(env)
    return _finite_reward((history.lifted & ~history.release_ready).float() * alignment)


def transport_reward(env: ManagerBasedRLEnv, std: float = 0.30) -> torch.Tensor:
    local, _, _, _, _ = _placement_values(env)
    distance = torch.linalg.vector_norm(local - local.new_tensor((0.0, 0.0, 0.0355)), dim=-1)
    history = _history(env)
    active = history.lifted & ~history.release_ready
    return _finite_reward(active.float() * (1.0 - torch.tanh(distance / std)))


def insertion_joint_goal_reward(env: ManagerBasedRLEnv, std: float = 0.5) -> torch.Tensor:
    """Guide the five arm joints toward the verified fixed-rack insertion pose."""
    robot: Articulation = env.scene["robot"]
    joint_ids = robot.find_joints(
        ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        preserve_order=True,
    )[0]
    position = _tensor(robot.data.joint_pos)[:, joint_ids]
    distance = torch.linalg.vector_norm(position - position.new_tensor(INSERTION_JOINT_GOAL), dim=-1)
    active = _history(env).lifted & ~_history(env).release_ready
    return _finite_reward(active.float() * (1.0 - torch.tanh(distance / std)))


class JointGoalProgressReward(ManagerTermBase):
    """One-step progress toward the verified insertion joint configuration."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_distance = torch.zeros(self.num_envs, device=self.device)
        self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        robot: Articulation = env.scene["robot"]
        self._joint_ids = robot.find_joints(
            ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            preserve_order=True,
        )[0]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._previous_distance[ids] = 0.0
        self._initialized[ids] = False

    def __call__(self, env: ManagerBasedRLEnv, scale: float = 0.05) -> torch.Tensor:
        robot: Articulation = env.scene["robot"]
        position = _tensor(robot.data.joint_pos)[:, self._joint_ids]
        distance = torch.linalg.vector_norm(position - position.new_tensor(INSERTION_JOINT_GOAL), dim=-1)
        progress = (self._previous_distance - distance) / scale
        progress = torch.where(self._initialized, progress, torch.zeros_like(progress))
        self._previous_distance.copy_(distance)
        self._initialized.fill_(True)
        active = _history(env).lifted & ~_history(env).release_ready
        return _finite_reward(active.float() * progress.clamp(-1.0, 1.0))


class TransportProgressReward(ManagerTermBase):
    """Potential-difference reward for immediate rack-directed transport.

    The persistent joint target makes each policy action influence many later
    states.  Absolute distance shaping alone can therefore have a long credit
    horizon.  This term pays signed one-step progress, while keeping its
    previous-distance buffer instance-owned and safe for partial resets.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_distance = torch.zeros(self.num_envs, device=self.device)
        self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._previous_distance[ids] = 0.0
        self._initialized[ids] = False

    def __call__(self, env: ManagerBasedRLEnv, scale: float = 0.01) -> torch.Tensor:
        local, _, _, _, _ = _placement_values(env)
        target = local.new_tensor((0.0, 0.0, 0.0355))
        distance = torch.linalg.vector_norm(local - target, dim=-1)
        progress = (self._previous_distance - distance) / scale
        progress = torch.where(self._initialized, progress, torch.zeros_like(progress))
        self._previous_distance.copy_(distance)
        self._initialized.fill_(True)
        history = _history(env)
        active = history.lifted & ~history.release_ready
        return _finite_reward(active.float() * progress.clamp(-1.0, 1.0))


def hand_transport_reward(env: ManagerBasedRLEnv, std: float = 0.20) -> torch.Tensor:
    """Guide the grasp frame toward the rack's in-hand insertion pose."""
    rack: RigidObject = env.scene["rack"]
    local = rack_local_position(
        _grasp_center_w(env),
        _tensor(rack.data.root_pos_w),
        _tensor(rack.data.root_quat_w),
    )
    distance = torch.linalg.vector_norm(local - local.new_tensor((0.0, 0.0, 0.055)), dim=-1)
    _, _, _, _, placed = _placement_values(env)
    active = _history(env).lifted & retained_grasp(env) & ~placed & ~_history(env).release_ready
    return _finite_reward(active.float() * (1.0 - torch.tanh(distance / std)))


def rack_insertion_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward reaching a valid rack pose before asking the hand to release."""
    _, alignment, _, _, placed = _placement_values(env)
    history = _history(env)
    return _finite_reward((history.lifted & ~history.release_ready & placed).float() * alignment)


def insertion_depth_reward(env: ManagerBasedRLEnv, std: float = 0.012) -> torch.Tensor:
    """Guide a grasped vial to the low, centered pose required for release."""
    local, alignment, _, _, placed = _placement_values(env)
    target = local.new_tensor((0.0, 0.0, 0.0355))
    distance = torch.linalg.vector_norm(local - target, dim=-1)
    history = _history(env)
    active = history.lifted & placed & retained_grasp(env) & ~history.release_ready
    return _finite_reward(active.float() * alignment * (1.0 - torch.tanh(distance / std)))


def _gripper_openness(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    gripper_id = robot.find_joints("gripper", preserve_order=True)[0]
    position = _tensor(robot.data.joint_pos)[:, gripper_id].squeeze(1)
    limits = _tensor(robot.data.soft_joint_pos_limits)[:, gripper_id].squeeze(1)
    return _finite_reward(((position - limits[:, 0]) / (limits[:, 1] - limits[:, 0])).clamp(0.0, 1.0))


def _release_pose_quality(env: ManagerBasedRLEnv) -> torch.Tensor:
    _, alignment, speed, _, placed = _placement_values(env)
    speed_quality = 1.0 - torch.tanh(speed / 0.10)
    return _finite_reward(placed.float() * alignment * speed_quality)


def release_shaping_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Open only after the vial has settled in the rack while still grasped."""
    # Preserve a gradient across the full joint range.  Saturating at 15%
    # openness made a barely separated jaw as valuable as a safely retracted
    # one and allowed exploratory contacts to re-grasp the vial.
    return _history(env).release_ready.float() * _release_pose_quality(env) * _gripper_openness(env)


def release_action_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Give immediate signed credit to the gripper delta after insertion."""
    action = env.action_manager.action[:, -1].clamp(-1.0, 1.0)
    return _history(env).release_ready.float() * action


def released_settle_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Keep a release-ready vial slow, upright, and within the rack bounds."""
    _, _, _, released, _ = _placement_values(env)
    return (_history(env).release_ready & released).float() * _release_pose_quality(env)


def premature_release_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Keep the jaw closed between first grasp and the settled rack pose."""
    history = _history(env)
    _, alignment, _, _, placed = _placement_values(env)
    may_release = history.release_ready & placed & (alignment > 0.8)
    excessive_opening = torch.relu(_gripper_openness(env) - 0.15)
    return (history.grasped & ~may_release).float() * excessive_opening


def gripper_hold_error(env: ManagerBasedRLEnv, target: float = -0.02) -> torch.Tensor:
    """Keep the jaw at the calibrated bilateral-contact setpoint during carry."""
    robot: Articulation = env.scene["robot"]
    gripper_id = robot.find_joints("gripper", preserve_order=True)[0]
    position = _tensor(robot.data.joint_pos)[:, gripper_id].squeeze(1)
    active = _history(env).grasped & ~_history(env).release_ready
    return _finite_reward(active.float() * torch.square(position - target))


def placement_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    _, alignment, speed, released, placed = _placement_values(env)
    gate = _history(env).lifted & placed & (alignment > 0.8)
    return (gate & released & (speed < 0.1)).float()


def success_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a one-step terminal bonus after stable placement confirmation."""
    return _history(env).success.float()


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _finite_reward(torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1))


def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize persistent joint-target drift from unnecessary commands."""
    return _finite_reward(torch.sum(torch.square(env.action_manager.action), dim=1))


def joint_velocity_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    return _finite_reward(torch.sum(torch.square(_tensor(robot.data.joint_vel)), dim=1))


def drop_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    vial: RigidObject = env.scene["vial"]
    return (_history(env).lifted & (_tensor(vial.data.root_pos_w)[:, 2] < 0.015)).float()


def workspace_exit_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize leaving the reachable tabletop workspace before termination."""
    vial: RigidObject = env.scene["vial"]
    root_pos_w, root_quat_w = _robot_root_pose(env)
    pos, _ = subtract_frame_transforms(root_pos_w, root_quat_w, _tensor(vial.data.root_pos_w))
    return ((pos[:, 2] < 0.0) | (pos[:, 0].abs() > 0.55) | (pos[:, 1].abs() > 0.45)).float()


def failure_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize task-ending object loss or non-finite robot motion.

    Early termination already removes future return, but an explicit terminal
    cost prevents a policy from learning to drop the vial as a shortcut out of
    the harder transport portion of an episode.
    """
    return (vial_lost(env) | unstable_robot(env)).float()


def vial_lost(env: ManagerBasedRLEnv) -> torch.Tensor:
    vial: RigidObject = env.scene["vial"]
    root_pos_w, root_quat_w = _robot_root_pose(env)
    vial_pos = _tensor(vial.data.root_pos_w)
    pos, _ = subtract_frame_transforms(root_pos_w, root_quat_w, vial_pos)
    finite = torch.isfinite(vial_pos).all(dim=-1) & torch.isfinite(_tensor(vial.data.root_quat_w)).all(dim=-1)
    finite &= torch.isfinite(_tensor(vial.data.root_lin_vel_w)).all(dim=-1)
    finite &= torch.isfinite(_tensor(vial.data.root_ang_vel_w)).all(dim=-1)
    history = _history(env)
    _, _, _, released, placed = _placement_values(env)
    dropped = history.lifted & (pos[:, 2] < 0.015)
    deliberate_bad_release = history.lifted & released & (_gripper_openness(env) > 0.15) & ~placed
    return (
        (~finite)
        | dropped
        | deliberate_bad_release
        | (pos[:, 2] < -0.01)
        | (pos[:, 0].abs() > 0.65)
        | (pos[:, 1].abs() > 0.55)
    )


def unstable_robot(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    joint_pos = _tensor(robot.data.joint_pos)
    joint_vel = _tensor(robot.data.joint_vel)
    finite = torch.isfinite(joint_pos).all(dim=-1) & torch.isfinite(joint_vel).all(dim=-1)
    return (~finite) | (joint_vel.abs().amax(dim=1) > 15.0)
