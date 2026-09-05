"""Observations, rewards, milestones, and terminations for the vial-placement task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, subtract_frame_transforms

from isaaclab_tutorial.tasks.place_vial.mdp.geometry import (
    cylinder_lowest_offset,
    inside_bounds,
    rack_local_position,
    symmetric_axial_keypoint_error,
    vertical_alignment,
)
from isaaclab_tutorial.tasks.place_vial.mdp.progress import PlacementProgress

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import Camera, ContactSensor


# ---------------------------------------------------------------------------------------------------------------------
# Task geometry (metres). All rack quantities are expressed in the rack frame, whose origin is the target opening.
# ---------------------------------------------------------------------------------------------------------------------

# The mat is centred at 32 mm and is 6 mm thick; the horizontal vial has a 17 mm collision radius.
VIAL_REST_HEIGHT = 0.052
# A released vial seats at about 31 mm; the success box accepts the seated pose, not a vial resting on the rim.
RACK_LOWER = (-0.015, -0.015, 0.026)
RACK_UPPER = (0.015, 0.015, 0.040)
# Held insertion ends with the vial root near 60 mm so the jaws stay clear of the rack; gravity finishes the travel.
HELD_INSERTION_TARGET = (0.0, 0.0, 0.060)
RACK_RIM_HEIGHT = 0.073
RACK_CLEARANCE_HEIGHT = RACK_RIM_HEIGHT + 0.008
# The 48 mm opening leaves the 34 mm vial about 7 mm of play per axis. A tip below the rim within this radius is in
# the target opening; the neighbouring openings are 60 mm away.
INSERTION_RADIUS = 0.012
# Cosine of the vial axis with vertical; 0.9 is about 26 degrees of tilt.
UPRIGHT_ALIGNMENT = 0.90
VIAL_AXIS_MIN = -0.017
VIAL_AXIS_MAX = 0.100
VIAL_RADIUS = 0.017
# The workshop grasp encloses the enlarged cap and its shoulder, which retains the vial axially.
VIAL_GRASP_OFFSET = (0.0, 0.0, 0.092)
# Both jaws touching a vial that is held this far above its resting height is a load-bearing grasp.
GRASP_PROOF_LIFT = 0.006
# Light rack guidance is expected during a real insertion; only much larger forces are counted as impacts.
HARD_RACK_IMPACT_FORCE = 20.0
FIXED_FINGERTIP_OFFSET = (-0.01289, -0.00022, -0.0986)
MOVING_FINGERTIP_OFFSET = (-0.0068, -0.0766, 0.0189)


def _tensor(value):
    """Return the torch view of an Isaac Lab proxy array."""
    return value.torch if hasattr(value, "torch") else value


def _finite(value: torch.Tensor) -> torch.Tensor:
    """Replace invalid terminal-state values with finite neutral values."""
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def _finite_error(value: torch.Tensor) -> torch.Tensor:
    """Replace invalid geometric errors with a safely distant value."""
    return torch.nan_to_num(value, nan=1.0e3, posinf=1.0e3, neginf=1.0e3)


# ---------------------------------------------------------------------------------------------------------------------
# Physical state helpers
# ---------------------------------------------------------------------------------------------------------------------


def _contact_magnitude(env: ManagerBasedRLEnv, name: str) -> torch.Tensor:
    """Return the maximum filtered contact-force magnitude for a sensor."""
    sensor: ContactSensor = env.scene.sensors[name]
    forces = _tensor(sensor.data.force_matrix_w)
    magnitude = torch.linalg.vector_norm(forces, dim=-1).reshape(env.num_envs, -1).amax(dim=-1)
    return _finite(magnitude)


def _contact(env: ManagerBasedRLEnv, name: str, threshold: float = 0.05) -> torch.Tensor:
    """Return thresholded filtered contact for one sensor."""
    return _contact_magnitude(env, name) > threshold


def bilateral_contact(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Whether the vial has simultaneous physical contact with both jaws."""
    return _contact(env, "fixed_jaw_contact") & _contact(env, "moving_jaw_contact")


def hard_rack_impact(force: torch.Tensor, threshold: float = HARD_RACK_IMPACT_FORCE) -> torch.Tensor:
    """Classify only contact forces large enough to be a physical impact."""
    return force > threshold


def unsafe_rack_contact(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Report hard vial/rack impacts while permitting ordinary guidance."""
    return hard_rack_impact(_contact_magnitude(env, "vial_rack_contact"))


def vial_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the vial root height above its environment origin [m]."""
    vial: RigidObject = env.scene["vial"]
    return _tensor(vial.data.root_pos_w)[:, 2] - env.scene.env_origins[:, 2]


def vial_lowest_height_in_rack(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the vial's lowest physical point in rack-local Z [m]."""
    vial: RigidObject = env.scene["vial"]
    rack: RigidObject = env.scene["rack"]
    vial_pos = _tensor(vial.data.root_pos_w)
    vial_quat = _tensor(vial.data.root_quat_w)
    rack_pos = _tensor(rack.data.root_pos_w)
    rack_quat = _tensor(rack.data.root_quat_w)
    root_r = rack_local_position(vial_pos, rack_pos, rack_quat)
    local_axis = vial_pos.new_tensor((0.0, 0.0, 1.0)).expand_as(vial_pos)
    axis_r = quat_apply_inverse(rack_quat, quat_apply(vial_quat, local_axis))
    return root_r[:, 2] + cylinder_lowest_offset(axis_r[:, 2], VIAL_AXIS_MIN, VIAL_AXIS_MAX, VIAL_RADIUS)


def fingertip_positions_w(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Return both simplified contact-pad centres in world coordinates [m]."""
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


def grasp_center_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the midpoint between the fingertip contact pads [m]."""
    fixed, moving = fingertip_positions_w(env)
    return 0.5 * (fixed + moving)


def vial_grasp_point_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the centre of the vial cap/shoulder used for grasping [m]."""
    vial: RigidObject = env.scene["vial"]
    root_pos = _tensor(vial.data.root_pos_w)
    root_quat = _tensor(vial.data.root_quat_w)
    offset = root_pos.new_tensor(VIAL_GRASP_OFFSET).expand_as(root_pos)
    return root_pos + quat_apply(root_quat, offset)


def _gripper_openness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return normalized measured jaw opening in [0, 1]."""
    robot: Articulation = env.scene["robot"]
    gripper_id = robot.find_joints("gripper", preserve_order=True)[0]
    position = _tensor(robot.data.joint_pos)[:, gripper_id].squeeze(1)
    limits = _tensor(robot.data.soft_joint_pos_limits)[:, gripper_id].squeeze(1)
    return _finite(((position - limits[:, 0]) / (limits[:, 1] - limits[:, 0])).clamp(0.0, 1.0))


def _placement_values(env: ManagerBasedRLEnv):
    """Return rack-local pose quality and physical release state."""
    vial: RigidObject = env.scene["vial"]
    rack: RigidObject = env.scene["rack"]
    vial_pos = _tensor(vial.data.root_pos_w)
    vial_quat = _tensor(vial.data.root_quat_w)
    local = rack_local_position(vial_pos, _tensor(rack.data.root_pos_w), _tensor(rack.data.root_quat_w))
    local = torch.nan_to_num(local, nan=1.0e3, posinf=1.0e3, neginf=-1.0e3)
    alignment = _finite(vertical_alignment(vial_quat))
    linear_speed = _finite_error(torch.linalg.vector_norm(_tensor(vial.data.root_lin_vel_w), dim=-1))
    angular_speed = _finite_error(torch.linalg.vector_norm(_tensor(vial.data.root_ang_vel_w), dim=-1))
    touching = _contact(env, "fixed_jaw_contact") | _contact(env, "moving_jaw_contact")
    released = ~touching & (_gripper_openness(env) > 0.20)
    placed = inside_bounds(local, RACK_LOWER, RACK_UPPER)
    return local, alignment, linear_speed, angular_speed, released, placed


def vial_inserted(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Whether the vial's tip is inside the target rack opening with the vial upright.

    The uprightness condition (the same one success uses) matters for the teacher's release: without it the state
    policy learns to let a tilted vial drop in from above, which works with perfect state but is a knife edge for a
    camera student to imitate.
    """
    local, alignment, *_ = _placement_values(env)
    centred = torch.linalg.vector_norm(local[:, :2], dim=-1) < INSERTION_RADIUS
    return centred & (vial_lowest_height_in_rack(env) < RACK_RIM_HEIGHT) & (alignment > UPRIGHT_ALIGNMENT)


# ---------------------------------------------------------------------------------------------------------------------
# Milestones, success, and terminations
# ---------------------------------------------------------------------------------------------------------------------


def _history(env: ManagerBasedRLEnv) -> PlacementProgress:
    """Return the instance-owned episode progress buffer."""
    progress = getattr(env, "_so101_placement_progress", None)
    if progress is None:
        progress = PlacementProgress(env.num_envs, env.device, stable_steps=10, grasp_steps=2)
        env._so101_placement_progress = progress
    return progress


class PlacementHistoryTerm(ManagerTermBase):
    """Latch physical milestones each step and terminate on confirmed placement success.

    Success requires the released vial to rest upright inside the target opening for ten consecutive control steps.
    Milestones seeded by downstream resets are restored on reset so they are never rewarded merely for being loaded.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.progress = _history(env)
        self._max_rack_force = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.progress.reset(env_ids)
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        )
        self._max_rack_force[ids] = 0.0
        grasped = getattr(self._env, "_so101_reset_grasped", None)
        lifted = getattr(self._env, "_so101_reset_lifted", None)
        if grasped is not None:
            self.progress.grasped[ids] = grasped[ids]
            self.progress.grasp_count[ids] = torch.where(grasped[ids], self.progress.grasp_steps, 0)
        if lifted is not None:
            self.progress.lifted[ids] = lifted[ids]

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        local, alignment, speed, angular_speed, released, placed = _placement_values(env)
        holding = bilateral_contact(env) & (vial_height(env) > VIAL_REST_HEIGHT + GRASP_PROOF_LIFT)
        cleared = vial_lowest_height_in_rack(env) >= RACK_CLEARANCE_HEIGHT
        inserted = vial_inserted(env)
        seated = placed & (alignment > UPRIGHT_ALIGNMENT) & released & (speed < 0.06) & (angular_speed < 0.8)
        rack_force = _contact_magnitude(env, "vial_rack_contact")
        self._max_rack_force.copy_(torch.maximum(self._max_rack_force, rack_force))
        self.progress.unsafe_rack_contact |= hard_rack_impact(rack_force)
        success = self.progress.update(holding, cleared, inserted, seated, env.episode_length_buf)

        log = env.extras.setdefault("log", {})
        log["Metrics/grasp_rate"] = self.progress.grasped.float().mean()
        log["Metrics/lift_rate"] = self.progress.lifted.float().mean()
        log["Metrics/insertion_rate"] = self.progress.inserted.float().mean()
        log["Metrics/seated_rate"] = placed.float().mean()
        log["Metrics/success_rate"] = success.float().mean()
        log["Safety/unsafe_rack_contact_rate"] = self.progress.unsafe_rack_contact.float().mean()
        log["Diagnostics/vial_height_m"] = _finite(vial_height(env)).mean()
        log["Diagnostics/vertical_alignment"] = alignment.mean()
        log["Diagnostics/gripper_openness"] = _gripper_openness(env).mean()
        log["Diagnostics/rack_local_z_m"] = _finite(local[:, 2]).mean()
        completed = self.progress.time_to_success >= 0
        if completed.any():
            log["Metrics/time_to_success_s"] = self.progress.time_to_success[completed].float().mean() * env.step_dt
        phase = getattr(env, "_so101_reset_phase", None)
        if phase is not None:
            log["Reset/mean_phase"] = phase.float().mean()

        # Terminal statistics survive the manager reset that follows, so evaluation hooks can read them.
        env._so101_terminal_progress = torch.stack(
            (self.progress.grasped, self.progress.lifted, self.progress.inserted, self.progress.unsafe_rack_contact),
            dim=-1,
        ).clone()
        env._so101_terminal_max_rack_force = self._max_rack_force.clone()
        env._so101_terminal_time_to_success_s = self.progress.time_to_success.clone().float() * env.step_dt
        return success


def vial_lost(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate non-finite, dropped, or unreachable vial states."""
    vial: RigidObject = env.scene["vial"]
    root_pos_w, root_quat_w = _robot_root_pose(env)
    vial_pos = _tensor(vial.data.root_pos_w)
    pos, _ = subtract_frame_transforms(root_pos_w, root_quat_w, vial_pos)
    finite = torch.isfinite(vial_pos).all(dim=-1) & torch.isfinite(_tensor(vial.data.root_quat_w)).all(dim=-1)
    finite &= torch.isfinite(_tensor(vial.data.root_lin_vel_w)).all(dim=-1)
    finite &= torch.isfinite(_tensor(vial.data.root_ang_vel_w)).all(dim=-1)
    dropped = _history(env).lifted & (pos[:, 2] < 0.004)
    return (~finite) | dropped | (pos[:, 2] < -0.01) | (pos[:, 0].abs() > 0.62) | (pos[:, 1].abs() > 0.50)


def unstable_robot(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate non-finite or extreme joint motion."""
    robot: Articulation = env.scene["robot"]
    joint_pos = _tensor(robot.data.joint_pos)
    joint_vel = _tensor(robot.data.joint_vel)
    finite = torch.isfinite(joint_pos).all(dim=-1) & torch.isfinite(joint_vel).all(dim=-1)
    return (~finite) | (joint_vel.abs().amax(dim=1) > 12.0)


# ---------------------------------------------------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------------------------------------------------


def held_object_goal_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the symmetry-aware vial pose error to the held insertion pose [m]."""
    vial: RigidObject = env.scene["vial"]
    rack: RigidObject = env.scene["rack"]
    rack_quat = _tensor(rack.data.root_quat_w)
    vial_position = rack_local_position(_tensor(vial.data.root_pos_w), _tensor(rack.data.root_pos_w), rack_quat)
    vial_axis_w = quat_apply(
        _tensor(vial.data.root_quat_w), vial_position.new_tensor((0.0, 0.0, 1.0)).expand_as(vial_position)
    )
    error = symmetric_axial_keypoint_error(
        vial_position,
        quat_apply_inverse(rack_quat, vial_axis_w),
        vial_position.new_tensor(HELD_INSERTION_TARGET).expand_as(vial_position),
        vial_position.new_tensor((0.0, 0.0, 1.0)).expand_as(vial_position),
        VIAL_AXIS_MIN,
        VIAL_AXIS_MAX,
    )
    return _finite_error(error)


class ApproachProgressReward(ManagerTermBase):
    """Pay the decrease in jaw-to-cap distance, in units of ``scale``, until the vial is grasped.

    A shaping term expressed as a difference of potentials has no incentive to hover: the total paid over an approach
    equals the distance covered. It also supplies the same gradient 25 cm from the vial as 2 cm away, which is what
    lets a pixels-only policy discover the approach from the home pose; a Gaussian bump around the cap is flat there.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_distance = torch.zeros(self.num_envs, device=self.device)
        self._has_previous = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._has_previous[ids] = False

    def __call__(self, env: ManagerBasedRLEnv, scale: float = 0.01) -> torch.Tensor:
        distance = _finite_error(torch.linalg.vector_norm(grasp_center_w(env) - vial_grasp_point_w(env), dim=-1))
        progress = ((self._previous_distance - distance) / scale).clamp(-1.0, 1.0)
        active = self._has_previous & ~_history(env).grasped
        self._previous_distance.copy_(distance)
        self._has_previous.fill_(True)
        return torch.where(active, progress, torch.zeros_like(progress))


def held_goal_reward(env: ManagerBasedRLEnv, goal_std: float = 0.10) -> torch.Tensor:
    """Dense shaping that brings the held vial to the insertion pose.

    Active only while the vial is grasped and its tip is not yet in the opening, so opening the jaws is the only
    profitable continuation after insertion. There is deliberately no equivalent bump around the vial before the grasp:
    a policy whose grasp is still unreliable learns to hover in such a bump instead of attempting the grasp.
    """
    history = _history(env)
    goal = torch.exp(-held_object_goal_error(env) / goal_std)
    held = history.grasped & ~history.inserted
    return _finite(torch.where(held, goal, torch.zeros_like(goal)))


class PhysicalMilestoneReward(ManagerTermBase):
    """Pay each physical milestone (grasped, lifted, inserted) exactly once per episode.

    Milestones already present on the first step after a reset are never paid, so downstream resets are not
    rewarded merely for being loaded.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous = torch.zeros((self.num_envs, 3), dtype=torch.bool, device=self.device)
        self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._previous[ids] = False
        self._initialized[ids] = False

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        history = _history(env)
        current = torch.stack((history.grasped, history.lifted, history.inserted), dim=-1)
        newly_reached = current & ~self._previous
        weights = current.new_tensor((1.0, 2.0, 4.0), dtype=torch.float32)
        reward = (newly_reached.float() * weights).sum(dim=-1)
        reward = torch.where(self._initialized, reward, torch.zeros_like(reward))
        self._previous.copy_(current)
        self._initialized.fill_(True)
        return _finite(reward)


def success_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the one-step confirmed success signal."""
    return _history(env).success.float()


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize changes in policy command."""
    return _finite(torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1))


def joint_velocity_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize high articulation velocity."""
    robot: Articulation = env.scene["robot"]
    return _finite(torch.sum(torch.square(_tensor(robot.data.joint_vel)), dim=1))


# ---------------------------------------------------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------------------------------------------------


def progress_flags(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the latched milestone flags (grasped, lifted, inserted) that make the reward Markov."""
    history = _history(env)
    return torch.stack((history.grasped, history.lifted, history.inserted), dim=-1).float()


def contact_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return privileged fixed-jaw, moving-jaw, and bilateral contact flags."""
    fixed = _contact(env, "fixed_jaw_contact")
    moving = _contact(env, "moving_jaw_contact")
    return torch.stack((fixed, moving, fixed & moving), dim=-1).float()


def joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Return finite measured joint positions for the selected robot joints."""
    asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    robot: Articulation = env.scene[asset_cfg.name]
    position = _finite(_tensor(robot.data.joint_pos)[:, asset_cfg.joint_ids])
    limits = _tensor(robot.data.soft_joint_pos_limits)[:, asset_cfg.joint_ids]
    return position.clamp(limits[..., 0], limits[..., 1])


def joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Return finite measured joint velocities for the selected robot joints."""
    asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    robot: Articulation = env.scene[asset_cfg.name]
    return _finite(_tensor(robot.data.joint_vel)[:, asset_cfg.joint_ids]).clamp(-12.0, 12.0)


def joint_target(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the six joint-position targets sent to the drives."""
    robot: Articulation = env.scene["robot"]
    return _finite(_tensor(robot.data.joint_pos_target)).clamp(-4.0, 4.0)


def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the previous finite policy action."""
    return _finite(env.action_manager.action).clamp(-1.0, 1.0)


def _robot_root_pose(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the fixed robot base pose used as the task reference frame."""
    robot: Articulation = env.scene["robot"]
    return _tensor(robot.data.root_link_pos_w), _tensor(robot.data.root_link_quat_w)


def body_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return robot-body pose and velocity in the robot base frame."""
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
    state = torch.cat((body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b), dim=-1).reshape(env.num_envs, -1)
    return _finite(state).clamp(-10.0, 10.0)


def rigid_object_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return rigid-object pose and velocity in the robot base frame."""
    obj: RigidObject = env.scene[asset_cfg.name]
    root_pos_w, root_quat_w = _robot_root_pose(env)
    obj_pos_b, obj_quat_b = subtract_frame_transforms(
        root_pos_w, root_quat_w, _tensor(obj.data.root_pos_w), _tensor(obj.data.root_quat_w)
    )
    state = torch.cat(
        (
            obj_pos_b,
            obj_quat_b,
            quat_apply_inverse(root_quat_w, _tensor(obj.data.root_lin_vel_w)),
            quat_apply_inverse(root_quat_w, _tensor(obj.data.root_ang_vel_w)),
        ),
        dim=-1,
    )
    return _finite(state).clamp(-10.0, 10.0)


def rack_relative_target(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the vial position in the rack frame [m]."""
    return _finite(_placement_values(env)[0]).clamp(-1.0, 1.0)


def placement_features(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the nonlinear insertion geometry a compact actor would otherwise have to derive from raw poses."""
    local, alignment, linear_speed, _, _, _ = _placement_values(env)
    xy_distance = torch.linalg.vector_norm(local[:, :2], dim=-1)
    return torch.stack((xy_distance, vial_lowest_height_in_rack(env), alignment, linear_speed), dim=-1).clamp(-1, 1)


class DomainRandomizedCameraImage(ManagerTermBase):
    """Read normalized wrist RGB with episode-consistent exposure, contrast, white-balance, and brightness variation.

    Isaac Lab's play mode disables observation corruption, in which case this term returns the rendered image.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._exposure_range = self._validate_range("exposure_range", cfg.params["exposure_range"], positive=True)
        self._contrast_range = self._validate_range("contrast_range", cfg.params["contrast_range"], positive=True)
        self._white_balance_range = self._validate_range(
            "white_balance_range", cfg.params["white_balance_range"], positive=True
        )
        self._brightness_range = self._validate_range("brightness_range", cfg.params["brightness_range"])
        shape = (self.num_envs, 1, 1, 1)
        self._exposure = torch.ones(shape, device=self.device)
        self._contrast = torch.ones(shape, device=self.device)
        self._white_balance = torch.ones((self.num_envs, 3, 1, 1), device=self.device)
        self._brightness = torch.zeros(shape, device=self.device)
        self.reset()

    @staticmethod
    def _validate_range(name: str, values: tuple[float, float], *, positive: bool = False) -> tuple[float, float]:
        lower, upper = values
        if not lower <= upper or (positive and lower <= 0.0):
            qualifier = "positive and ordered" if positive else "ordered"
            raise ValueError(f"{name} must be {qualifier}, got {values}.")
        return float(lower), float(upper)

    @staticmethod
    def _resample(tensor: torch.Tensor, env_ids: Sequence[int] | None, value_range: tuple[float, float]) -> None:
        if env_ids is None:
            tensor.uniform_(*value_range)
        else:
            tensor[env_ids] = torch.empty_like(tensor[env_ids]).uniform_(*value_range)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._resample(self._exposure, env_ids, self._exposure_range)
        self._resample(self._contrast, env_ids, self._contrast_range)
        self._resample(self._white_balance, env_ids, self._white_balance_range)
        self._resample(self._brightness, env_ids, self._brightness_range)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        exposure_range: tuple[float, float],
        contrast_range: tuple[float, float],
        white_balance_range: tuple[float, float],
        brightness_range: tuple[float, float],
    ) -> torch.Tensor:
        camera: Camera = env.scene.sensors[sensor_cfg.name]
        image = _tensor(camera.data.output["rgb"])[..., :3]
        image = image.permute(0, 3, 1, 2).contiguous().float().div(255.0)
        if env.cfg.observations.wrist_rgb.enable_corruption:
            image = (image - 0.5) * self._contrast + 0.5
            image = (image * self._exposure * self._white_balance + self._brightness).clamp(0.0, 1.0)
        return image
