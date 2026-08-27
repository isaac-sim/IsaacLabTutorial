"""Observations, physical task rewards, history, and terminations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, subtract_frame_transforms

from .geometry import (
    cylinder_lowest_offset,
    inside_bounds,
    rack_local_position,
    symmetric_axial_keypoint_error,
    vertical_alignment,
)
from .progress import PlacementProgress

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import Camera, ContactSensor


# The workshop mat is centered at 32 mm, is 6 mm thick, and the horizontal
# vial has a 17 mm collision radius: 0.032 + 0.003 + 0.017 = 0.052 m.
VIAL_REST_HEIGHT = 0.052
# The held insertion ends above 30 mm so the jaws remain clear of the rack;
# after release, gravity seats the vial at approximately 31 mm in the rack
# frame. This is measured from centered free-drop trials using the rack
# and is substantially below the 68 mm rim-engagement pose. The target is the
# released pose, never a command for the robot to push through the rack.
RACK_TARGET = (0.0, 0.0, 0.031)
# The gripper cannot follow the vial to the released seat without intersecting
# the rack.  Physics-validated phase-six endpoints place the held root near
# 60 mm, with its lower tip already inside the opening; gravity completes the
# remaining travel after the jaw opens.
HELD_INSERTION_TARGET = (0.0, 0.0, 0.060)
# A successful released vial must be near its mechanically seated height, not
# merely engaged with the top of the opening. Held insertion is tracked by
# ``held_insertion_ready`` and deliberately uses a separate, higher region.
RACK_LOWER = (-0.015, -0.015, 0.026)
RACK_UPPER = (0.015, 0.015, 0.040)
RACK_RIM_HEIGHT = 0.073
RACK_CLEARANCE_MARGIN = 0.008
RACK_CLEARANCE_HEIGHT = RACK_RIM_HEIGHT + RACK_CLEARANCE_MARGIN
# At the workshop tabletop pose the horizontal vial's lowest point is 5 mm
# below the rack frame origin: 52 mm root height - 40 mm rack height - 17 mm
# radius. Lift progress is measured from this physical surface, not the root.
TABLETOP_LOWEST_HEIGHT_IN_RACK = -0.005
RACK_XY_BOUNDS = (-0.031, 0.091, -0.031, 0.091)
VIAL_AXIS_MIN = -0.017
VIAL_AXIS_MAX = 0.100
VIAL_RADIUS = 0.017
# The workshop grasp encloses the enlarged cap and its shoulder. This provides
# geometric axial retention under the identified finite-force gripper drive;
# a smooth-body grasp made contact but failed the independent proof lift.
VIAL_GRASP_OFFSET = (0.0, 0.0, 0.092)
# The reset generator accepts a proof grasp only when the jaw midpoint stays
# within this distance of the intended cap/shoulder grasp point.  Use the same
# physical criterion for the online milestone so a bilateral tip touch is not
# mistaken for a load-bearing grasp.
GRASP_CENTER_TOLERANCE = 0.015
# Closing is much less forgiving than retaining an established grasp.  The
# jaw midpoint must first descend to within a few millimeters of the generated
# grasp point; the broader tolerance above remains appropriate after contact
# while the vial rotates in hand.
GRASP_PROOF_LIFT = 0.006
# Light rack guidance is expected during a real insertion.  Only contact well
# beyond the force needed to guide this 20 g vial is classified as an impact
# for diagnostics and reset-generation rejection.
HARD_RACK_IMPACT_FORCE = 20.0
# The vial can rotate a few degrees while gravity transfers it from the jaws
# to the hole.  Require a modest upright margin before authorizing release so
# a marginal 0.90 pose cannot immediately settle below the 0.90 success bound.
HELD_INSERTION_ALIGNMENT = 0.985
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
    axis_w = quat_apply(vial_quat, local_axis)
    axis_r = quat_apply_inverse(rack_quat, axis_w)
    return root_r[:, 2] + cylinder_lowest_offset(axis_r[:, 2], VIAL_AXIS_MIN, VIAL_AXIS_MAX, VIAL_RADIUS)


def lift_clearance_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return lift progress based on the vial's lowest physical point."""
    travel = RACK_CLEARANCE_HEIGHT - TABLETOP_LOWEST_HEIGHT_IN_RACK
    progress = (vial_lowest_height_in_rack(env) - TABLETOP_LOWEST_HEIGHT_IN_RACK) / travel
    return _finite(progress).clamp(0.0, 1.0)


def lift_clearance_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Densify the existing rack-clearance milestone while the grasp is live."""
    history = _history(env)
    active = ~history.lifted & load_bearing_grasp(env)
    travel = RACK_CLEARANCE_HEIGHT - TABLETOP_LOWEST_HEIGHT_IN_RACK
    grasp_height = vial_grasp_point_w(env)[:, 2] - env.scene.env_origins[:, 2]
    reference = getattr(env, "_so101_grasp_reference_height", None)
    if reference is None:
        upward_progress = torch.zeros_like(grasp_height)
    else:
        upward_progress = ((grasp_height - reference) / travel).clamp(0.0, 1.0)
    progress = torch.maximum(lift_clearance_progress(env), upward_progress)
    return _finite(active.float() * progress)


class LoadBearingLiftProgressReward(ManagerTermBase):
    """Reward only upward progress while the vial is physically load-bearing."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous = torch.zeros(self.num_envs, device=self.device)
        self._previous_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._previous[ids] = 0.0
        self._previous_active[ids] = False

    def __call__(self, env: ManagerBasedRLEnv, scale: float = 0.01) -> torch.Tensor:
        if scale <= 0.0:
            raise ValueError("scale must be positive.")
        travel = RACK_CLEARANCE_HEIGHT - TABLETOP_LOWEST_HEIGHT_IN_RACK
        grasp_height = vial_grasp_point_w(env)[:, 2] - env.scene.env_origins[:, 2]
        reference = getattr(env, "_so101_grasp_reference_height", None)
        if reference is None:
            progress = torch.zeros_like(grasp_height)
        else:
            progress = ((grasp_height - reference) / travel).clamp(0.0, 1.0)
            progress = torch.maximum(progress, lift_clearance_progress(env))
        active = ~_history(env).lifted & load_bearing_grasp(env)
        progress = _finite(progress)
        reward = ((progress - self._previous) / scale).clamp(-1.0, 1.0)
        reward = torch.where(active & self._previous_active, reward, torch.zeros_like(reward))
        self._previous.copy_(progress)
        self._previous_active.copy_(active)
        return _finite(reward)


def rack_clearance_violation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return unsafe rack-overflight states before the vial clears the rim.

    The vial may descend only through the selected hole. Everywhere else it
    must first raise its lowest point above the rack's top lattice.
    """
    local = _placement_values(env)[0]
    x_min, x_max, y_min, y_max = RACK_XY_BOUNDS
    over_rack = (local[:, 0] > x_min) & (local[:, 0] < x_max)
    over_rack &= (local[:, 1] > y_min) & (local[:, 1] < y_max)
    in_target_hole = torch.linalg.vector_norm(local[:, :2], dim=-1) < 0.012
    below_clearance = vial_lowest_height_in_rack(env) < RACK_CLEARANCE_HEIGHT
    return (bilateral_contact(env) & over_rack & below_clearance & ~in_target_hole).float()


def undesired_rack_contact(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a soft force cost, reduced for intended insertion guidance."""
    local, alignment, *_ = _placement_values(env)
    insertion_corridor = (torch.linalg.vector_norm(local[:, :2], dim=-1) < 0.014) & (alignment > 0.88)
    force = (_contact_magnitude(env, "vial_rack_contact") / HARD_RACK_IMPACT_FORCE).clamp(0.0, 1.0)
    corridor_scale = torch.where(insertion_corridor, 0.1, 1.0)
    return force * corridor_scale


def hard_rack_impact(force: torch.Tensor, threshold: float = HARD_RACK_IMPACT_FORCE) -> torch.Tensor:
    """Classify only contact forces large enough to be a physical impact."""
    return force > threshold


def unsafe_rack_contact(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Report hard vial/rack impacts while permitting ordinary guidance."""
    return hard_rack_impact(_contact_magnitude(env, "vial_rack_contact"))


def fingertip_positions_w(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Return both simplified contact-pad centers in world coordinates [m]."""
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
    """Return the center of the workshop vial body used for grasping [m]."""
    vial: RigidObject = env.scene["vial"]
    root_pos = _tensor(vial.data.root_pos_w)
    root_quat = _tensor(vial.data.root_quat_w)
    offset = root_pos.new_tensor(VIAL_GRASP_OFFSET).expand_as(root_pos)
    return root_pos + quat_apply(root_quat, offset)


def load_bearing_grasp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Whether both jaws hold the vial near its validated grasp point."""
    distance = torch.linalg.vector_norm(grasp_center_w(env) - vial_grasp_point_w(env), dim=-1)
    return bilateral_contact(env) & (distance < GRASP_CENTER_TOLERANCE)


def grasp_proof_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return progress toward lifting the grasp point 6 mm from reset."""
    height = vial_grasp_point_w(env)[:, 2] - env.scene.env_origins[:, 2]
    reference = getattr(env, "_so101_grasp_reference_height", None)
    if reference is None:
        return torch.zeros_like(height)
    return ((height - reference) / GRASP_PROOF_LIFT).clamp(0.0, 1.0)


def grasp_proof_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Densify only the physical 6 mm load-bearing grasp proof."""
    unproven = ~_history(env).grasped
    return unproven.float() * load_bearing_grasp(env).float() * grasp_proof_progress(env)


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
    linear_speed = torch.nan_to_num(
        torch.linalg.vector_norm(_tensor(vial.data.root_lin_vel_w), dim=-1),
        nan=1.0e3,
        posinf=1.0e3,
        neginf=1.0e3,
    )
    angular_speed = torch.nan_to_num(
        torch.linalg.vector_norm(_tensor(vial.data.root_ang_vel_w), dim=-1),
        nan=1.0e3,
        posinf=1.0e3,
        neginf=1.0e3,
    )
    touching = _contact(env, "fixed_jaw_contact") | _contact(env, "moving_jaw_contact")
    released = ~touching & (_gripper_openness(env) > 0.20)
    placed = inside_bounds(local, RACK_LOWER, RACK_UPPER)
    return local, alignment, linear_speed, angular_speed, released, placed


def held_insertion_ready(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Whether a held vial is safely engaged with the selected rack opening."""
    local, alignment, speed, _, _, _ = _placement_values(env)
    # The detailed opening has only a few millimeters of radial clearance.
    # A broad rack-level bound authorizes release onto a rail and reproduces
    # the high, visibly incomplete placements this term is meant to prevent.
    centered = torch.linalg.vector_norm(local[:, :2], dim=-1) < 0.004
    tip_inside_opening = vial_lowest_height_in_rack(env) < RACK_RIM_HEIGHT
    # The jaws cannot follow the vial to its seated root pose without striking
    # the rack. A real placement therefore releases after the vial tip enters
    # the opening, then lets gravity seat it while the hand stays outside.
    above_seated_pose = local[:, 2] > 0.030
    return centered & tip_inside_opening & above_seated_pose & (alignment > HELD_INSERTION_ALIGNMENT) & (speed < 0.12)


def _history(env: ManagerBasedRLEnv) -> PlacementProgress:
    """Return the instance-owned episode progress buffer."""
    progress = getattr(env, "_so101_placement_progress", None)
    if progress is None:
        progress = PlacementProgress(env.num_envs, env.device, stable_steps=10, grasp_steps=2)
        env._so101_placement_progress = progress
    return progress


class PlacementHistoryTerm(ManagerTermBase):
    """Confirm success from physical state with safe partial-reset history."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.progress = _history(env)
        self._max_rack_force = torch.zeros(self.num_envs, device=self.device)
        self._grasp_reference_height = torch.zeros(self.num_envs, device=self.device)
        env._so101_grasp_reference_height = self._grasp_reference_height

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset selected rows and seed milestones represented by reset data."""
        self.progress.reset(env_ids)
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        )
        self._max_rack_force[ids] = 0.0
        grasp_height = vial_grasp_point_w(self._env)[:, 2] - self._env.scene.env_origins[:, 2]
        self._grasp_reference_height[ids] = grasp_height[ids]
        grasped = getattr(self._env, "_so101_reset_grasped", None)
        lifted = getattr(self._env, "_so101_reset_lifted", None)
        if grasped is not None:
            self.progress.grasped[ids] = grasped[ids]
            self.progress.grasp_count[ids] = torch.where(
                grasped[ids],
                torch.full_like(self.progress.grasp_count[ids], self.progress.grasp_steps),
                torch.zeros_like(self.progress.grasp_count[ids]),
            )
        if lifted is not None:
            self.progress.lifted[ids] = lifted[ids]
        phase = getattr(self._env, "_so101_reset_phase", None)
        if phase is not None:
            # Release rows are validated continuations of a held insertion.
            self.progress.release_ready[ids] = phase[ids] >= 7

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Update milestones and return stable released placement success."""
        vial: RigidObject = env.scene["vial"]
        vial_z = _tensor(vial.data.root_pos_w)[:, 2]
        local, alignment, speed, angular_speed, released, placed = _placement_values(env)
        contact = bilateral_contact(env)
        # Centering is an acquisition criterion.  Once a load-bearing grasp
        # has been proved (or seeded by a validated downstream reset), normal
        # bilateral contact is sufficient while the vial rotates in-hand.
        # A live, centered bilateral hold that raises the vial by 6 mm proves
        # load bearing. Do not constrain that proof to the reset position or
        # initial vial orientation: translating and beginning reorientation
        # during the proof are both valid physical behavior.
        proof = load_bearing_grasp(env) & (grasp_proof_progress(env) >= 1.0)
        proof &= (speed < 0.08) & (angular_speed < 0.8)
        proof &= vial_z - env.scene.env_origins[:, 2] > 0.038
        grasp = torch.where(self.progress.grasped, contact, proof)
        insertion_ready = held_insertion_ready(env)
        valid_grasped = insertion_ready & grasp
        rack_force = _contact_magnitude(env, "vial_rack_contact")
        self._max_rack_force.copy_(torch.maximum(self._max_rack_force, rack_force))
        unsafe_contact = hard_rack_impact(rack_force)
        self.progress.unsafe_rack_contact |= unsafe_contact
        valid_released = placed & (alignment > 0.90) & released & (speed < 0.06) & (angular_speed < 0.8)
        success = self.progress.update(
            grasp,
            lift_clearance_progress(env) >= 1.0,
            valid_released,
            env.episode_length_buf,
            valid_grasped,
        )
        log = env.extras.setdefault("log", {})
        log["Metrics/grasp_rate"] = self.progress.grasped.float().mean()
        log["Metrics/live_bilateral_grasp_rate"] = grasp.float().mean()
        log["Metrics/lift_rate"] = self.progress.lifted.float().mean()
        log["Metrics/insertion_rate"] = insertion_ready.float().mean()
        log["Metrics/seated_rate"] = placed.float().mean()
        log["Metrics/release_rate"] = (self.progress.lifted & placed & released).float().mean()
        log["Metrics/success_rate"] = success.float().mean()
        log["Safety/unsafe_rack_contact_rate"] = self.progress.unsafe_rack_contact.float().mean()
        # Failed environments can contain one terminal physics sample before
        # their reset is applied. Keep summaries useful without hiding the
        # non-finite state from ``vial_lost`` below.
        log["Diagnostics/vial_height_m"] = _finite(vial_z).mean()
        log["Diagnostics/vertical_alignment"] = _finite(alignment).mean()
        log["Diagnostics/gripper_openness"] = _gripper_openness(env).mean()
        log["Diagnostics/gripper_action"] = _finite(env.action_manager.action[:, -1]).mean()
        log["Diagnostics/rack_local_x_m"] = _finite(local[:, 0]).mean()
        log["Diagnostics/rack_local_y_m"] = _finite(local[:, 1]).mean()
        log["Diagnostics/rack_local_z_m"] = _finite(local[:, 2]).mean()
        phase = getattr(env, "_so101_reset_phase", None)
        if phase is not None:
            log["Reset/mean_phase"] = phase.float().mean()
        completed = self.progress.time_to_success >= 0
        if completed.any():
            log["Metrics/time_to_success_s"] = self.progress.time_to_success[completed].float().mean() * env.step_dt
        # Keep the pre-reset milestones available to external evaluation
        # callbacks. Manager resets happen before the wrapped step returns.
        env._so101_terminal_progress = torch.stack(
            (
                self.progress.grasped,
                self.progress.lifted,
                self.progress.release_ready,
                self.progress.unsafe_rack_contact,
            ),
            dim=-1,
        ).clone()
        env._so101_terminal_max_rack_force = self._max_rack_force.clone()
        env._so101_terminal_time_to_success_s = self.progress.time_to_success.clone().float() * env.step_dt
        env._so101_terminal_insertion_state = torch.stack(
            (
                local[:, 0],
                local[:, 1],
                local[:, 2],
                alignment,
                speed,
                vial_lowest_height_in_rack(env),
            ),
            dim=-1,
        ).clone()
        if phase is not None:
            env._so101_terminal_reset_phase = phase.clone()
        return success


class HeldInsertionHistoryTerm(PlacementHistoryTerm):
    """Maintain the full physical history while ending at held insertion."""

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        super().__call__(env)
        return self.progress.release_ready


def progress_flags(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return privileged irreversible milestone flags for the critic."""
    history = _history(env)
    return torch.stack((history.grasped, history.lifted, history.release_ready), dim=-1).float()


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
    """Return the six ordinary joint-position targets sent to the USD drives."""
    robot: Articulation = env.scene["robot"]
    return _finite(_tensor(robot.data.joint_pos_target)).clamp(-4.0, 4.0)


def last_action(env: ManagerBasedRLEnv, action_name: str | None = None) -> torch.Tensor:
    """Return the previous finite policy action."""
    action = env.action_manager.action if action_name is None else env.action_manager.get_term(action_name).raw_actions
    return _finite(action).clamp(-1.0, 1.0)


def _robot_root_pose(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the fixed robot base pose used as task reference frame."""
    robot: Articulation = env.scene["robot"]
    return _tensor(robot.data.root_link_pos_w), _tensor(robot.data.root_link_quat_w)


def body_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return robot-body state in the robot base frame."""
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
    """Return rigid-object state in the robot base frame."""
    obj: RigidObject = env.scene[asset_cfg.name]
    root_pos_w, root_quat_w = _robot_root_pose(env)
    obj_pos_b, obj_quat_b = subtract_frame_transforms(
        root_pos_w,
        root_quat_w,
        _tensor(obj.data.root_pos_w),
        _tensor(obj.data.root_quat_w),
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
    """Return vial position in the rack frame [m]."""
    return _finite(_placement_values(env)[0]).clamp(-1.0, 1.0)


def placement_features(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return continuous physical geometry useful near release.

    These values contain no phase, milestone, success, or contact label. They
    simply expose the nonlinear geometry already present in the raw poses so a
    compact actor can distinguish transport from a seated insertion.
    """
    local, alignment, linear_speed, _, _, _ = _placement_values(env)
    xy_distance = torch.linalg.vector_norm(local[:, :2], dim=-1)
    lowest_height = vial_lowest_height_in_rack(env)
    return torch.stack((xy_distance, lowest_height, alignment, linear_speed), dim=-1).clamp(-1.0, 1.0)


def visual_geometry_target(env: ManagerBasedRLEnv, position_scale: float = 0.25) -> torch.Tensor:
    """Return compact task geometry in the wrist/gripper frame for auxiliary learning.

    The deployed actor never consumes this group. During simulation training it
    gives the visual encoder a direct localization signal: the vial grasp point,
    the held-insertion target, and the vial axis, all expressed in the camera's
    fixed parent-link frame. Positions are normalized by the workspace scale.
    """
    if position_scale <= 0.0:
        raise ValueError("position_scale must be positive.")
    robot: Articulation = env.scene["robot"]
    gripper_id = robot.find_bodies("gripper", preserve_order=True)[0]
    gripper_pos_w = _tensor(robot.data.body_pos_w)[:, gripper_id].squeeze(1)
    gripper_quat_w = _tensor(robot.data.body_quat_w)[:, gripper_id].squeeze(1)

    vial: RigidObject = env.scene["vial"]
    vial_quat_w = _tensor(vial.data.root_quat_w)
    vial_point_g, _ = subtract_frame_transforms(
        gripper_pos_w,
        gripper_quat_w,
        vial_grasp_point_w(env),
        vial_quat_w,
    )

    rack: RigidObject = env.scene["rack"]
    rack_pos_w = _tensor(rack.data.root_pos_w)
    rack_quat_w = _tensor(rack.data.root_quat_w)
    insertion_offset = rack_pos_w.new_tensor(HELD_INSERTION_TARGET).expand_as(rack_pos_w)
    insertion_pos_w = rack_pos_w + quat_apply(rack_quat_w, insertion_offset)
    insertion_point_g, _ = subtract_frame_transforms(
        gripper_pos_w,
        gripper_quat_w,
        insertion_pos_w,
        rack_quat_w,
    )

    vial_axis_w = quat_apply(vial_quat_w, vial_quat_w.new_tensor((0.0, 0.0, 1.0)).expand_as(gripper_pos_w))
    vial_axis_g = quat_apply_inverse(gripper_quat_w, vial_axis_w)
    target = torch.cat((vial_point_g / position_scale, insertion_point_g / position_scale, vial_axis_g), dim=-1)
    return _finite(target).clamp(-2.0, 2.0)


class DomainRandomizedCameraImage(ManagerTermBase):
    """Read normalized wrist RGB with episode-consistent sensor variation.

    Exposure, contrast, and white balance are sampled independently for each
    environment at reset.  The variation models ordinary camera/illumination
    changes without replacing scene geometry or giving the actor privileged
    state.  Isaac Lab's play mode disables observation corruption, in which
    case this term returns the asset-authored render unchanged.
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
            return
        values = torch.empty_like(tensor[env_ids]).uniform_(*value_range)
        tensor[env_ids] = values

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


class TemporalDomainRandomizedCameraImage(DomainRandomizedCameraImage):
    """Return a short RGB history without changing camera resolution."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        self._history_length = int(cfg.params["history_length"])
        if self._history_length < 2:
            raise ValueError("Temporal wrist observations require at least two frames.")
        self._frame_history: torch.Tensor | None = None
        self._history_initialized: torch.Tensor | None = None
        self._last_common_step = -1
        super().__init__(cfg, env)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if self._history_initialized is None:
            return
        if env_ids is None:
            self._history_initialized.zero_()
        else:
            self._history_initialized[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        exposure_range: tuple[float, float],
        contrast_range: tuple[float, float],
        white_balance_range: tuple[float, float],
        brightness_range: tuple[float, float],
        history_length: int,
    ) -> torch.Tensor:
        del history_length
        current = super().__call__(
            env,
            sensor_cfg,
            exposure_range,
            contrast_range,
            white_balance_range,
            brightness_range,
        )
        if self._frame_history is None:
            self._frame_history = current[:, None].repeat(1, self._history_length, 1, 1, 1)
            self._history_initialized = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self._last_common_step = int(env.common_step_counter)
        else:
            assert self._history_initialized is not None
            uninitialized = ~self._history_initialized
            if uninitialized.any():
                self._frame_history[uninitialized] = current[uninitialized, None]
                self._history_initialized[uninitialized] = True
            common_step = int(env.common_step_counter)
            if common_step != self._last_common_step:
                initialized = ~uninitialized
                self._frame_history[initialized, :-1] = self._frame_history[initialized, 1:].clone()
                self._frame_history[initialized, -1] = current[initialized]
                self._last_common_step = common_step
        return self._frame_history.flatten(1, 2)


def reaching_reward(env: ManagerBasedRLEnv, std: float = 0.08) -> torch.Tensor:
    """Reward bringing the jaw midpoint to the vial's body center."""
    distance = torch.linalg.vector_norm(grasp_center_w(env) - vial_grasp_point_w(env), dim=-1)
    return _finite((~_history(env).grasped).float() * (1.0 - torch.tanh(distance / std)))


def held_object_goal_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the symmetry-aware vial error at the final held pose."""
    vial: RigidObject = env.scene["vial"]
    rack: RigidObject = env.scene["rack"]
    vial_position = rack_local_position(
        _tensor(vial.data.root_pos_w),
        _tensor(rack.data.root_pos_w),
        _tensor(rack.data.root_quat_w),
    )
    vial_axis_w = quat_apply(
        _tensor(vial.data.root_quat_w),
        vial_position.new_tensor((0.0, 0.0, 1.0)).expand_as(vial_position),
    )
    vial_axis = quat_apply_inverse(_tensor(rack.data.root_quat_w), vial_axis_w)
    error = symmetric_axial_keypoint_error(
        vial_position,
        vial_axis,
        vial_position.new_tensor(HELD_INSERTION_TARGET).expand_as(vial_position),
        vial_position.new_tensor((0.0, 0.0, 1.0)).expand_as(vial_position),
        VIAL_AXIS_MIN,
        VIAL_AXIS_MAX,
    )
    # Invalid terminal samples must never become an accidental perfect score.
    return _finite_error(error)


def held_object_goal_error_cost(env: ManagerBasedRLEnv, max_error: float = 0.3) -> torch.Tensor:
    """Penalize remaining final-pose error only while a lifted vial is held."""
    if max_error <= 0.0:
        raise ValueError("max_error must be positive.")
    active = _history(env).lifted & ~_history(env).release_ready
    return active.float() * held_object_goal_error(env).clamp_max(max_error)


def held_object_goal_basin_reward(env: ManagerBasedRLEnv, radius: float = 0.10) -> torch.Tensor:
    """Reward settling only inside a compact basin around the final held pose."""
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    history = _history(env)
    # Once insertion has been physically confirmed, remove the holding reward
    # permanently so opening and gravity seating are the profitable continuation.
    active = history.lifted & ~history.release_ready
    basin = (1.0 - held_object_goal_error(env) / radius).clamp(0.0, 1.0)
    return active.float() * basin


def object_goal_reward(
    env: ManagerBasedRLEnv,
    approach_std: float = 0.08,
    goal_std: float = 0.10,
    held_scale: float = 1.0,
    approach_opening_threshold: float = 0.0,
    approach_close_distance: float = 0.0,
    approach_close_bonus: float = 0.0,
    held_contact_bonus: float = 0.0,
    use_live_grasp_goal: bool = False,
    require_lift_for_goal: bool = False,
) -> torch.Tensor:
    """Shape approach before grasp and object pose while physically held.

    The held goal is the final insertion pose itself, expressed through the
    vial's center and axial endpoints. The endpoint assignment is symmetric,
    so no arbitrary vial yaw or signed-axis convention is imposed here.
    Physical milestones and final seating remain the directional authorities.
    """
    approach_distance = torch.linalg.vector_norm(grasp_center_w(env) - vial_grasp_point_w(env), dim=-1)
    approach = torch.exp(-approach_distance / approach_std)
    if approach_opening_threshold > 0.0:
        opening_gate = (_gripper_openness(env) / approach_opening_threshold).clamp(0.0, 1.0)
        if approach_close_distance > 0.0:
            close_bonus = 1.0 + approach_close_bonus * (1.0 - _gripper_openness(env))
            opening_gate = torch.where(
                approach_distance <= approach_close_distance,
                close_bonus,
                opening_gate,
            )
        approach = approach * opening_gate

    goal_error = held_object_goal_error(env)
    goal = held_scale * torch.exp(-goal_error / goal_std)

    grasped = _history(env).grasped
    held = grasped & ~_history(env).release_ready
    before_proof = approach + load_bearing_grasp(env).float() * goal if use_live_grasp_goal else approach
    held_goal = torch.where(_history(env).lifted, goal, torch.zeros_like(goal)) if require_lift_for_goal else goal
    held_goal = held_contact_bonus + held_goal
    reward = torch.where(~grasped, before_proof, torch.where(held, held_goal, torch.zeros_like(goal)))
    return _finite(reward)


class HeldObjectGoalProgressReward(ManagerTermBase):
    """Reward progress toward the one final held pose after physical lift."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_error = torch.zeros(self.num_envs, device=self.device)
        self._previous_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._previous_error[ids] = 0.0
        self._previous_active[ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        scale: float = 0.01,
        require_lift: bool = True,
    ) -> torch.Tensor:
        if scale <= 0.0:
            raise ValueError("scale must be positive.")
        error = held_object_goal_error(env)
        history = _history(env)
        proved = history.lifted if require_lift else history.grasped
        active = proved & ~history.release_ready
        error = _finite(error)
        reward = ((self._previous_error - error) / scale).clamp(-1.0, 1.0)
        reward = torch.where(active & self._previous_active, reward, torch.zeros_like(reward))
        self._previous_error.copy_(error)
        self._previous_active.copy_(active)
        return _finite(reward)


class HeldUprightProgressReward(ManagerTermBase):
    """Reward signed vial-uprighting progress after a physical lift."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_alignment = torch.zeros(self.num_envs, device=self.device)
        self._previous_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._previous_alignment[ids] = 0.0
        self._previous_active[ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        scale: float = 0.02,
        require_lift: bool = True,
    ) -> torch.Tensor:
        if scale <= 0.0:
            raise ValueError("scale must be positive.")
        alignment = _finite(_placement_values(env)[1])
        history = _history(env)
        proved = history.lifted if require_lift else history.grasped
        active = proved & ~history.release_ready
        reward = ((alignment - self._previous_alignment) / scale).clamp(-1.0, 1.0)
        reward = torch.where(active & self._previous_active, reward, torch.zeros_like(reward))
        self._previous_alignment.copy_(alignment)
        self._previous_active.copy_(active)
        return _finite(reward)


def held_upright_alignment_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward absolute vertical alignment only while the lifted vial is held."""
    history = _history(env)
    active = history.lifted & ~history.release_ready
    alignment = _finite(_placement_values(env)[1])
    return _finite(active.float() * alignment)


def held_upright_clearance_reward(env: ManagerBasedRLEnv, height_std: float = 0.01) -> torch.Tensor:
    """Reward upright alignment only to the extent that rack clearance is retained."""
    if height_std <= 0.0:
        raise ValueError("height_std must be positive.")
    history = _history(env)
    active = history.lifted & ~history.release_ready
    alignment = _finite(_placement_values(env)[1])
    deficit = _finite_error(RACK_CLEARANCE_HEIGHT - vial_lowest_height_in_rack(env)).clamp_min(0.0)
    return _finite(active.float() * alignment * torch.exp(-deficit / height_std))


def held_upright_lift_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward the physical conjunction of upright alignment and lift clearance.

    Unlike a narrow clearance barrier, the existing normalized lift progress
    supplies a useful gradient over the entire tabletop-to-rack travel and
    saturates exactly at safe transport clearance.
    """
    history = _history(env)
    active = history.lifted & ~history.release_ready
    alignment = _finite(_placement_values(env)[1])
    return _finite(active.float() * alignment * lift_clearance_progress(env))


def held_lift_clearance_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward retaining a lifted vial up to exact transport clearance."""
    history = _history(env)
    active = history.lifted & ~history.release_ready
    return _finite(active.float() * lift_clearance_progress(env))


class HeldRadialCenterProgressReward(ManagerTermBase):
    """Reward signed progress toward the rack opening's radial center."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_distance = torch.zeros(self.num_envs, device=self.device)
        self._previous_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._previous_distance[ids] = 0.0
        self._previous_active[ids] = False

    def __call__(self, env: ManagerBasedRLEnv, scale: float = 0.005) -> torch.Tensor:
        if scale <= 0.0:
            raise ValueError("scale must be positive.")
        local = _placement_values(env)[0]
        distance = _finite_error(torch.linalg.vector_norm(local[:, :2], dim=-1))
        history = _history(env)
        active = history.lifted & ~history.release_ready
        reward = ((self._previous_distance - distance) / scale).clamp(-1.0, 1.0)
        reward = torch.where(active & self._previous_active, reward, torch.zeros_like(reward))
        self._previous_distance.copy_(distance)
        self._previous_active.copy_(active)
        return _finite(reward)


def held_radial_center_reward(env: ManagerBasedRLEnv, std: float = 0.02) -> torch.Tensor:
    """Reward a lifted, held vial for remaining centered over the rack opening."""
    if std <= 0.0:
        raise ValueError("std must be positive.")
    local = _placement_values(env)[0]
    distance = _finite_error(torch.linalg.vector_norm(local[:, :2], dim=-1))
    history = _history(env)
    active = history.lifted & ~history.release_ready
    return _finite(active.float() * torch.exp(-distance / std))


def held_tip_inside_reward(env: ManagerBasedRLEnv, std: float = 0.005) -> torch.Tensor:
    """Smooth the physical held-insertion height gate without rewarding over-descent."""
    if std <= 0.0:
        raise ValueError("std must be positive.")
    excess_height = _finite_error(vial_lowest_height_in_rack(env) - RACK_RIM_HEIGHT).clamp_min(0.0)
    history = _history(env)
    active = history.lifted & ~history.release_ready
    return _finite(active.float() * torch.exp(-excess_height / std))


def held_insertion_gate_reward(
    env: ManagerBasedRLEnv,
    radial_std: float = 0.004,
    height_std: float = 0.005,
    alignment_std: float = 0.01,
) -> torch.Tensor:
    """Return a smooth conjunction of the three physical held-insertion gates."""
    if min(radial_std, height_std, alignment_std) <= 0.0:
        raise ValueError("Insertion gate scales must be positive.")
    local, alignment, _, _, _, _ = _placement_values(env)
    radial_excess = _finite_error(torch.linalg.vector_norm(local[:, :2], dim=-1) - 0.004).clamp_min(0.0)
    height_excess = _finite_error(vial_lowest_height_in_rack(env) - RACK_RIM_HEIGHT).clamp_min(0.0)
    alignment_deficit = _finite_error(HELD_INSERTION_ALIGNMENT - alignment).clamp_min(0.0)
    history = _history(env)
    active = history.lifted & ~history.release_ready
    score = torch.exp(-radial_excess / radial_std - height_excess / height_std - alignment_deficit / alignment_std)
    return _finite(active.float() * score)


def held_radial_error_cost(env: ManagerBasedRLEnv, scale: float = 0.02) -> torch.Tensor:
    """Return normalized held radial error with a constant useful gradient."""
    if scale <= 0.0:
        raise ValueError("scale must be positive.")
    local = _placement_values(env)[0]
    distance = _finite_error(torch.linalg.vector_norm(local[:, :2], dim=-1))
    history = _history(env)
    active = history.lifted & ~history.release_ready
    return _finite(active.float() * (distance / scale).clamp_max(2.0))


def held_clearance_error_cost(env: ManagerBasedRLEnv, scale: float = 0.02) -> torch.Tensor:
    """Penalize lowering a lifted vial below safe transport clearance.

    This is a one-sided constraint: it supplies no incentive to raise the vial
    farther once its lowest physical point clears the rack rim.
    """
    if scale <= 0.0:
        raise ValueError("scale must be positive.")
    deficit = _finite_error(RACK_CLEARANCE_HEIGHT - vial_lowest_height_in_rack(env)).clamp_min(0.0)
    history = _history(env)
    active = history.lifted & ~history.release_ready
    return _finite(active.float() * (deficit / scale).clamp_max(2.0))


class PhysicalMilestoneReward(ManagerTermBase):
    """Pay each physics-confirmed task milestone exactly once.

    Generated downstream resets seed the corresponding milestones, so they
    are never rewarded merely for being loaded. The reward contains no pose
    schedule, waypoint, contact avoidance rule, or action interpretation.
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
        current = torch.stack((history.grasped, history.lifted, history.release_ready), dim=-1)
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


def release_opening_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward measured jaw opening only after physical rack engagement."""
    return _history(env).release_ready.float() * _gripper_openness(env)


def release_action_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward the jaw command direction only after physical insertion is confirmed."""
    action = _finite(env.action_manager.action[:, -1]).clamp(-1.0, 1.0)
    return _history(env).release_ready.float() * action


class ReleaseOpeningProgressReward(ManagerTermBase):
    """Reward signed jaw-opening progress after a valid held insertion."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_opening = torch.zeros(self.num_envs, device=self.device)
        self._previous_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._previous_opening[ids] = 0.0
        self._previous_active[ids] = False

    def __call__(self, env: ManagerBasedRLEnv, scale: float = 0.02) -> torch.Tensor:
        if scale <= 0.0:
            raise ValueError("scale must be positive.")
        opening = _gripper_openness(env)
        # Once physical insertion has been confirmed, keep paying the opening
        # potential while contact transfers from the jaws to the rack.  Gating
        # on the instantaneous held geometry would remove the learning signal
        # precisely when the jaws begin to release the vial.
        active = _history(env).release_ready
        reward = ((opening - self._previous_opening) / scale).clamp(-1.0, 1.0)
        reward = torch.where(active & self._previous_active, reward, torch.zeros_like(reward))
        self._previous_opening.copy_(opening)
        self._previous_active.copy_(active)
        return _finite(reward)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize changes in policy command."""
    return _finite(torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1))


def action_magnitude_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize normalized command magnitude without interpreting any action component."""
    return _finite(torch.sum(torch.square(env.action_manager.action), dim=1))


def arm_action_magnitude_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize arm commands while leaving the final gripper command unconstrained."""
    return _finite(torch.sum(torch.square(env.action_manager.action[:, :-1]), dim=1))


def joint_velocity_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize high articulation velocity."""
    robot: Articulation = env.scene["robot"]
    return _finite(torch.sum(torch.square(_tensor(robot.data.joint_vel)), dim=1))


def joint_limit_margin_l2(env: ManagerBasedRLEnv, margin: float = 0.20) -> torch.Tensor:
    """Penalize only the outer margin of each normalized soft joint range."""
    if not 0.0 < margin < 1.0:
        raise ValueError("margin must lie strictly between zero and one.")
    robot: Articulation = env.scene["robot"]
    joint_pos = _tensor(robot.data.joint_pos)
    limits = _tensor(robot.data.soft_joint_pos_limits)
    center = 0.5 * (limits[..., 0] + limits[..., 1])
    half_range = 0.5 * (limits[..., 1] - limits[..., 0]).clamp_min(1.0e-6)
    normalized = ((joint_pos - center) / half_range).abs()
    margin_progress = ((normalized - (1.0 - margin)) / margin).clamp_min(0.0)
    return _finite(margin_progress.square().sum(dim=-1))


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
