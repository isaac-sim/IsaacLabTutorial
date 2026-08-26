"""Generate and inspect physics-validated reset poses for the vial task."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..assets import RESET_DATASET
from ..control import (
    GRASP_GRIPPER_POSITION,
    PREGRASP_GRIPPER_POSITION,
    RELEASE_GRIPPER_POSITION,
    TABLETOP_VIAL_HEADING_RANGE,
    TABLETOP_VIAL_POSITION,
    TABLETOP_VIAL_POSITION_HALF_RANGE,
    WORKSHOP_INITIAL_JOINT_POSITION,
    WORKSHOP_PREGRASP_JOINT_POSITION,
    WORKSHOP_TASK_WAYPOINTS,
)
from .dataset import PHASE_NAMES, load_reset_dataset, save_reset_dataset


@dataclass(frozen=True)
class GeneratorCfg:
    """Reset generator quotas and physical validation settings."""

    poses_per_phase: int = 128
    # Detailed rack collision geometry can overflow Newton's triangle-pair
    # buffer when 256 inserted/released candidates are validated together.
    # This affects only offline generation throughput, not task physics.
    batch_size: int = 128
    seed: int = 42
    articulation_settle_steps: int = 8
    grasp_close_steps: int = 90
    transition_steps: int = 120
    contact_settle_steps: int = 90
    max_attempts_per_phase: int = 32_768
    branch_seed_count: int = 32
    joint_noise: float = 0.025
    vial_position_half_range: tuple[float, float] = TABLETOP_VIAL_POSITION_HALF_RANGE
    contact_distance: float = 0.030
    ik_seeds: int = 64
    ik_iterations: int = 120
    ik_noise_std: float = 0.65
    ik_position_tolerance: float = 0.005
    ik_rotation_tolerance: float = math.radians(18.0)
    ik_joint_margin: float = 0.03
    min_grasp_pad_alignment: float = 0.85

    def __post_init__(self) -> None:
        for name in (
            "poses_per_phase",
            "batch_size",
            "articulation_settle_steps",
            "grasp_close_steps",
            "transition_steps",
            "contact_settle_steps",
            "max_attempts_per_phase",
            "branch_seed_count",
            "ik_seeds",
            "ik_iterations",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.max_attempts_per_phase < self.poses_per_phase:
            raise ValueError("max_attempts_per_phase must be at least poses_per_phase.")
        if not 0.0 <= self.min_grasp_pad_alignment <= 1.0:
            raise ValueError("min_grasp_pad_alignment must be between zero and one.")
        if self.ik_joint_margin < 0.0:
            raise ValueError("ik_joint_margin must be non-negative.")
        if len(self.vial_position_half_range) != 2 or any(value <= 0.0 for value in self.vial_position_half_range):
            raise ValueError("vial_position_half_range must contain two positive half-widths.")


# Match the workshop task's authored tabletop orientation.  Yaw
# randomization is applied around this +90-degree pitch.
_HORIZONTAL_QUATERNION = (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))
PREGRASP_PROOF_DIFFICULTY = (1.0 + 2.0 / 3.0) / 7.0
LIFT_SEGMENT_VERTICAL_TRAVEL = 0.105


def measured_lift_progress(vial_height: torch.Tensor, initial_height: torch.Tensor) -> torch.Tensor:
    """Map measured root-height gain onto the physical lift segment."""
    return ((vial_height - initial_height) / LIFT_SEGMENT_VERTICAL_TRAVEL).clamp(0.0, 1.0)


def phase_quotas(cfg: GeneratorCfg) -> tuple[int, ...]:
    """Return the exact balanced phase quotas."""
    return (cfg.poses_per_phase,) * len(PHASE_NAMES)


def _tensor(value):
    return value.torch if hasattr(value, "torch") else value


def _represented_lift(
    phase: int,
    vial_pose: torch.Tensor,
    rack_z: torch.Tensor,
    represented_grasp: torch.Tensor,
) -> torch.Tensor:
    """Seed lift history only after the vial physically cleared the rack."""
    from ..mdp.geometry import cylinder_lowest_offset, quat_rotate_xyzw
    from ..mdp.terms import (
        RACK_CLEARANCE_HEIGHT,
        VIAL_AXIS_MAX,
        VIAL_AXIS_MIN,
        VIAL_RADIUS,
    )

    axis = quat_rotate_xyzw(
        vial_pose[:, 3:7],
        vial_pose.new_tensor((0.0, 0.0, 1.0)).expand(vial_pose.shape[0], -1),
    )
    lowest_in_rack = (
        vial_pose[:, 2] - rack_z + cylinder_lowest_offset(axis[:, 2], VIAL_AXIS_MIN, VIAL_AXIS_MAX, VIAL_RADIUS)
    )
    lifted_now = lowest_in_rack >= RACK_CLEARANCE_HEIGHT
    # Every later phase is connected through the terminal lift state. Its
    # current pose may descend again, but the milestone is irreversible.
    historically_lifted = phase >= 4
    return represented_grasp & (lifted_now | historically_lifted)


class _Generator:
    """Build candidates in simulation and keep only stable physical states."""

    def __init__(self, env, cfg: GeneratorCfg):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(env.device)
        self.random = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self.robot = env.scene["robot"]
        self.vial = env.scene["vial"]
        self.rack = env.scene["rack"]
        self.num_envs = env.num_envs
        self.gripper_body_id = self.robot.find_bodies("gripper", preserve_order=True)[0]
        self.moving_body_id = self.robot.find_bodies("moving_jaw_so101_v1", preserve_order=True)[0]
        self.action_term = env.action_manager.get_term("joint_delta")
        self._zeros = torch.zeros((self.num_envs, 6), device=self.device)
        self.rejections = {name: 0 for name in PHASE_NAMES}
        self.last_diagnostics: dict[str, float] = {}
        self.candidate_diagnostics: dict[str, float] = {}
        self.candidate_difficulty = torch.zeros(self.num_envs, device=self.device)
        self.candidate_ik_valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.candidate_unsafe = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.candidate_rack_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.candidate_clearance_violation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._track_safety = False
        self._track_clearance = False
        self._seed_banks: dict[str, dict[str, torch.Tensor]] = {}
        self._target_vial_quaternion = torch.zeros((self.num_envs, 4), device=self.device)
        self._target_vial_quaternion[:, 3] = 1.0
        # Environment construction has not run a reset yet, so its live
        # gripper coordinate may still be zero rather than the configured
        # open command. Pose the articulation explicitly before measuring the
        # aperture center used by the IK objective.
        open_geometry_position = _tensor(self.robot.data.default_joint_pos).clone()
        open_geometry_position[:, -1] = PREGRASP_GRIPPER_POSITION
        self._write_robot(open_geometry_position)
        self.env.scene.write_data_to_sim()
        self.env.sim.forward()
        self.env.scene.update(self.env.physics_dt)
        self.grasp_tcp_offset = self._pregrasp_center_offset()
        self._build_ik()
        # This is the normal close command observed on the real robot, not a
        # mechanical-limit command and not an actuator or contact override.
        self.gripper_closed_position = GRASP_GRIPPER_POSITION
        gripper_joint_id = self.robot.find_joints("gripper", preserve_order=True)[0][0]
        lower, upper = self.ik_joint_limits[gripper_joint_id].tolist()
        if not lower <= self.gripper_closed_position <= upper:
            raise RuntimeError(
                f"The real-robot grasp command {self.gripper_closed_position:.7f} rad is outside "
                f"the USD soft joint limits [{lower:.7f}, {upper:.7f}]."
            )

    def _pregrasp_center_offset(self) -> torch.Tensor:
        """Measure the collision-pad midpoint in the pregrasp frame.

        This is derived from the loaded robot geometry at runtime. Centering
        the open jaw aperture on the vial prevents either finger from sweeping
        the tabletop object during the validated approach.
        """
        from isaaclab.utils.math import quat_apply_inverse

        from ..mdp.terms import grasp_center_w

        position = _tensor(self.robot.data.body_pos_w)[:, self.gripper_body_id].squeeze(1)
        quaternion = _tensor(self.robot.data.body_quat_w)[:, self.gripper_body_id].squeeze(1)
        offsets = quat_apply_inverse(quaternion, grasp_center_w(self.env) - position)
        if not bool(torch.isfinite(offsets).all()):
            raise RuntimeError("The USD-authored pregrasp geometry is not finite.")
        spread = (offsets - offsets[0]).abs().amax()
        if float(spread) > 1.0e-5:
            raise RuntimeError("Cloned environments disagree on the pregrasp contact geometry.")
        return offsets[0].clone()

    def _build_ik(self) -> None:
        """Build the same batched Newton IK machinery used by reference reset generators."""
        import isaaclab.sim as sim_utils
        import warp as wp
        from isaaclab import cloner
        from isaaclab_newton.cloner import copy_newton_clone_source
        from isaaclab_newton.ik import (
            NewtonIKJointLimitObjectiveCfg,
            NewtonIKPoseObjectiveCfg,
            NewtonIKSolver,
            NewtonIKSolverCfg,
        )

        plan = sim_utils.SimulationContext.instance().get_clone_plan()
        resolved = cloner.query.path_to_source(plan, self.robot.cfg.prim_path) if plan is not None else None
        if resolved is None:
            raise RuntimeError("Could not resolve the SO-101 clone-plan source for reset IK.")
        source = copy_newton_clone_source(resolved[0])
        origin = -self.env.scene.env_origins[0]
        prototype_xform = wp.transform(wp.vec3(*origin.tolist()), wp.quat_identity())
        import newton

        prototype = newton.ModelBuilder(up_axis=source.up_axis)
        prototype.add_builder(source, xform=prototype_xform)
        self.ik_model = prototype.finalize(device=str(self.device))

        body_names = [str(label).rsplit("/", 1)[-1] for label in self.ik_model.body_label]
        gripper_matches = [index for index, name in enumerate(body_names) if name == "gripper"]
        if len(gripper_matches) != 1:
            raise RuntimeError("Expected exactly one Newton IK body named 'gripper'.")
        self.ik_gripper_body_id = gripper_matches[0]

        joint_names = [str(label).rsplit("/", 1)[-1] for label in self.ik_model.joint_label]
        joint_q_start = wp.to_torch(self.ik_model.joint_q_start).to(device=self.device, dtype=torch.long)

        coordinate_ids = []
        for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"):
            matches = [index for index, joint_name in enumerate(joint_names) if joint_name == name]
            if len(matches) != 1:
                raise RuntimeError(f"Expected exactly one Newton IK joint named {name!r}.")
            coordinate_ids.append(int(joint_q_start[matches[0]].item()))
        self.ik_coordinate_ids = torch.tensor(coordinate_ids, device=self.device, dtype=torch.long)
        self.ik_joint_seed = wp.to_torch(self.ik_model.joint_q).to(self.device).repeat(self.num_envs, 1)
        self.ik_joint_limits = _tensor(self.robot.data.soft_joint_pos_limits)[0].clone()

        objective_name = "reset_vial_gripper"
        self.ik_solver = NewtonIKSolver(
            NewtonIKSolverCfg(
                optimizer="lm",
                jacobian_mode="analytic",
                sampler="gauss",
                n_seeds=self.cfg.ik_seeds,
                noise_std=self.cfg.ik_noise_std,
                iterations=self.cfg.ik_iterations,
                lambda_initial=0.1,
                rng_seed=self.cfg.seed,
            ),
            model=self.ik_model,
            num_envs=self.num_envs,
            device=str(self.device),
            objectives=[
                NewtonIKPoseObjectiveCfg(
                    body_name="gripper",
                    name=objective_name,
                    body_offset_pos=tuple(float(value) for value in self.grasp_tcp_offset),
                    position_weight=100.0,
                    # The arm has five task DOFs. Position dominates while the
                    # rotation term keeps the long vial offset controllable;
                    # grasp axis feasibility is checked explicitly below.
                    rotation_weight=1.0,
                ),
                NewtonIKJointLimitObjectiveCfg(weight=1.0),
            ],
            link_resolver=lambda _name: self.ik_gripper_body_id,
        )
        self.ik_pose_objective = self.ik_solver.objectives_by_name[objective_name]

    def _desired_vial_pose(self, phase: int) -> torch.Tensor:
        """Sample an upright, rack-centered object pose for transport or insertion."""
        rack_position = _tensor(self.rack.data.root_pos_w) - self.env.scene.env_origins
        pose = torch.zeros((self.num_envs, 7), device=self.device)
        pose[:, :3] = rack_position
        if phase == 5:
            pose[:, :2] += torch.empty((self.num_envs, 2), device=self.device).uniform_(
                -0.006, 0.006, generator=self.random
            )
            # Values are rack-local root heights. The rack itself is 40 mm
            # above the world origin, so adding 155--190 mm here needlessly
            # pushed the five-DoF arm beyond its accurate workspace. This band
            # remains at least 17 mm above the rail-clearance requirement.
            pose[:, 2] += torch.empty(self.num_envs, device=self.device).uniform_(0.115, 0.140, generator=self.random)
            # Preserve the upright orientation reached during the physical
            # reorientation. A vial is yaw-symmetric, so imposing a new random
            # yaw here only asks the wrist to spin for no task benefit.
            pose[:, 3:7] = self._target_vial_quaternion
        else:
            pose[:, :2] += torch.empty((self.num_envs, 2), device=self.device).uniform_(
                -0.0010, 0.0010, generator=self.random
            )
            # Insert enough of the body for the hole to guide the free vial
            # during release. The grasp stays on the cap: at these root
            # heights its center is still 144--152 mm above the rack frame,
            # leaving both fingertips well clear of the 73 mm rim.
            pose[:, 2] += torch.empty(self.num_envs, device=self.device).uniform_(0.052, 0.060, generator=self.random)
            pose[:, 3:7] = self._target_vial_quaternion
        return pose

    def _reorient_vial(
        self,
        target: torch.Tensor,
        progress: torch.Tensor,
        *,
        move_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rotate a center-grasped, lifted vial toward a reachable upright pose.

        The endpoint is solved once and approached through slow joint-position
        interpolation. The vial stays fully dynamic, and measured alignment
        and bilateral contact decide whether the trajectory is retained.
        """
        from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul

        from ..mdp.geometry import vertical_alignment
        from ..mdp.terms import bilateral_contact

        start_pose = torch.cat(
            (
                _tensor(self.vial.data.root_pos_w) - self.env.scene.env_origins,
                _tensor(self.vial.data.root_quat_w),
            ),
            dim=-1,
        ).clone()
        vial_axis = quat_apply(
            start_pose[:, 3:7],
            start_pose.new_tensor((0.0, 0.0, 1.0)).expand(self.num_envs, -1),
        )
        upright_axis = start_pose.new_tensor((0.0, 0.0, 1.0)).expand_as(vial_axis)
        correction_axis = torch.linalg.cross(vial_axis, upright_axis, dim=-1)
        correction_norm = torch.linalg.vector_norm(correction_axis, dim=-1, keepdim=True)
        fallback_axis = start_pose.new_tensor((1.0, 0.0, 0.0)).expand_as(correction_axis)
        correction_axis = torch.where(
            correction_norm > 1.0e-6,
            correction_axis / correction_norm.clamp_min(1.0e-6),
            fallback_axis,
        )
        correction_angle = torch.acos((vial_axis * upright_axis).sum(-1).clamp(-1.0, 1.0))
        base_upright_quaternion = quat_mul(
            quat_from_angle_axis(correction_angle, correction_axis),
            start_pose[:, 3:7],
        )
        if move_mask is None:
            move_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        connected_valid = self.candidate_ik_valid.clone()
        endpoint_pose = start_pose.clone()
        endpoint_target = target.clone()
        endpoint_found = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        target_height = torch.maximum(start_pose[:, 2], torch.full_like(start_pose[:, 2], 0.095))
        endpoint_pose[:, 2] = target_height
        # Pull slightly toward the robot and away from the rack while rotating.
        # Axial spin is task-irrelevant for the round vial, but changes which
        # wrist branch the five-DoF arm can reach.
        for spin in (0.0, 0.5 * math.pi, -0.5 * math.pi, math.pi):
            spin_angle = torch.full((self.num_envs,), spin, device=self.device)
            spin_quaternion = quat_from_angle_axis(spin_angle, upright_axis)
            option_quaternion = quat_mul(spin_quaternion, base_upright_quaternion)
            for offset_x, offset_y in (
                (-0.04, -0.04),
                (0.00, -0.04),
                (-0.07, -0.04),
                (-0.04, -0.07),
                (0.00, -0.07),
            ):
                option_pose = start_pose.clone()
                option_pose[:, 0] += offset_x
                option_pose[:, 1] += offset_y
                option_pose[:, 2] = target_height
                option_pose[:, 3:7] = option_quaternion
                self.candidate_ik_valid.fill_(True)
                option_target = self._solve_vial_pose(option_pose, target, position_tolerance=0.015)
                option_valid = connected_valid & self.candidate_ik_valid
                select = option_valid & ~endpoint_found
                endpoint_pose = torch.where(select.unsqueeze(-1), option_pose, endpoint_pose)
                endpoint_target = torch.where(select.unsqueeze(-1), option_target, endpoint_target)
                endpoint_found |= option_valid
                if bool((endpoint_found | ~connected_valid).all()):
                    break
            if bool((endpoint_found | ~connected_valid).all()):
                break
        connected_valid &= ~move_mask | endpoint_found
        self.candidate_diagnostics["reorient_endpoint_valid_rate"] = float(endpoint_found.float().mean())
        self._target_vial_quaternion.copy_(endpoint_pose[:, 3:7])

        # Exact intermediate object poses over-constrain a five-DoF arm.
        # Instead, slowly interpolate its ordinary position command to the
        # reachable endpoint. The mat supplies the passive pivot reaction and
        # measured physics—not interpolation—sets the vial trajectory.
        start_target = target.clone()
        steps_per_waypoint = self.cfg.transition_steps
        settle_steps = max(1, self.cfg.contact_settle_steps // 3)
        terminal_mask = connected_valid & move_mask & torch.isclose(progress, torch.ones_like(progress))
        # Repeating the endpoint gives the original finite-stiffness drives
        # time to converge under load without commanding an overshoot.
        for waypoint in (0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.000, 1.000, 1.000):
            waypoint_progress = torch.minimum(progress, torch.full_like(progress, waypoint))
            proposed = torch.lerp(start_target, endpoint_target, waypoint_progress.unsqueeze(-1))
            waypoint_valid = connected_valid
            update = connected_valid & move_mask
            target = torch.where(update.unsqueeze(-1), proposed, target)
            self._move_robot(target, steps_per_waypoint)
            self._simulate(settle_steps, target)
            if bool(terminal_mask.any()):
                label = f"reorient_{round(100 * waypoint):03d}_terminal"
                alignment = vertical_alignment(_tensor(self.vial.data.root_quat_w))[terminal_mask]
                joint_error = torch.linalg.vector_norm(
                    _tensor(self.robot.data.joint_pos) - target,
                    dim=-1,
                )[terminal_mask]
                self.candidate_diagnostics[f"{label}_ik_rate"] = float(waypoint_valid[terminal_mask].float().mean())
                self.candidate_diagnostics[f"{label}_alignment_mean"] = float(alignment.mean())
                self.candidate_diagnostics[f"{label}_bilateral_rate"] = float(
                    bilateral_contact(self.env)[terminal_mask].float().mean()
                )
                self.candidate_diagnostics[f"{label}_joint_error_mean_rad"] = float(joint_error.mean())
        self.candidate_diagnostics["reorient_terminal_alignment_mean"] = (
            float(vertical_alignment(_tensor(self.vial.data.root_quat_w))[terminal_mask].mean())
            if bool(terminal_mask.any())
            else 0.0
        )
        self.candidate_ik_valid.copy_(connected_valid)
        return target

    def _begin_supported_pivot(
        self,
        target: torch.Tensor,
        progress: torch.Tensor,
        *,
        rotation_fraction: float = 1.0,
        move_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Raise the capped end while the opposite end pivots on the mat."""
        from isaaclab.utils.math import (
            axis_angle_from_quat,
            quat_apply,
            quat_conjugate,
            quat_from_angle_axis,
            quat_mul,
        )

        from ..mdp.geometry import cylinder_lowest_offset
        from ..mdp.terms import VIAL_AXIS_MAX, VIAL_AXIS_MIN, VIAL_RADIUS, VIAL_REST_HEIGHT

        if move_mask is None:
            move_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        start_pose = torch.cat(
            (
                _tensor(self.vial.data.root_pos_w) - self.env.scene.env_origins,
                _tensor(self.vial.data.root_quat_w),
            ),
            dim=-1,
        ).clone()
        local_axis = start_pose.new_tensor((0.0, 0.0, 1.0)).expand(self.num_envs, -1)
        vial_axis = quat_apply(start_pose[:, 3:7], local_axis)
        upright_axis = start_pose.new_tensor((0.0, 0.0, 1.0)).expand_as(vial_axis)
        correction_axis = torch.linalg.cross(vial_axis, upright_axis, dim=-1)
        correction_norm = torch.linalg.vector_norm(correction_axis, dim=-1, keepdim=True)
        fallback_axis = start_pose.new_tensor((1.0, 0.0, 0.0)).expand_as(correction_axis)
        correction_axis = torch.where(
            correction_norm > 1.0e-6,
            correction_axis / correction_norm.clamp_min(1.0e-6),
            fallback_axis,
        )
        correction_angle = torch.acos((vial_axis * upright_axis).sum(-1).clamp(-1.0, 1.0))
        correction_angle *= rotation_fraction
        upright_quaternion = quat_mul(
            quat_from_angle_axis(correction_angle, correction_axis),
            start_pose[:, 3:7],
        )
        relative = quat_mul(upright_quaternion, quat_conjugate(start_pose[:, 3:7]))
        relative_axis_angle = axis_angle_from_quat(relative)
        relative_angle = torch.linalg.vector_norm(relative_axis_angle, dim=-1)
        relative_axis = relative_axis_angle / relative_angle.unsqueeze(-1).clamp_min(1.0e-6)
        relative_axis = torch.where((relative_angle > 1.0e-6).unsqueeze(-1), relative_axis, fallback_axis)

        base_offset = start_pose.new_tensor((0.0, 0.0, VIAL_AXIS_MIN)).expand(self.num_envs, -1)
        base_anchor = start_pose[:, :3] + quat_apply(start_pose[:, 3:7], base_offset)
        table_height = VIAL_REST_HEIGHT - VIAL_RADIUS
        source_valid = self.candidate_ik_valid.clone()
        reached_waypoint = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        settle_steps = max(1, self.cfg.contact_settle_steps // 6)
        terminal = move_mask & torch.isclose(progress, torch.ones_like(progress))
        for waypoint in (0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.000):
            waypoint_progress = torch.minimum(progress, torch.full_like(progress, waypoint))
            desired_pose = start_pose.clone()
            increment = quat_from_angle_axis(relative_angle * waypoint_progress, relative_axis)
            desired_pose[:, 3:7] = quat_mul(increment, start_pose[:, 3:7])
            desired_axis = quat_apply(desired_pose[:, 3:7], local_axis)
            desired_base_offset = quat_apply(desired_pose[:, 3:7], base_offset)
            desired_pose[:, :2] = base_anchor[:, :2] - desired_base_offset[:, :2]
            desired_pose[:, 2] = table_height - cylinder_lowest_offset(
                desired_axis[:, 2], VIAL_AXIS_MIN, VIAL_AXIS_MAX, VIAL_RADIUS
            )
            self.candidate_ik_valid.fill_(True)
            proposed = self._solve_vial_pose(
                desired_pose,
                target,
                position_tolerance=0.015,
                # The first supported-pivot waypoint moves the cap through a
                # roughly 20 mm arc. On this short five-DoF arm that ordinary
                # Cartesian move can require about one radian distributed
                # across the arm. Keep a generous discontinuity guard while
                # selecting the nearest valid IK branch below; 0.45 rad
                # incorrectly rejected every physically connected waypoint.
                max_joint_delta=1.5,
            )
            waypoint_valid = self.candidate_ik_valid.clone()
            update = source_valid & move_mask & waypoint_valid
            reached_waypoint |= update
            target = torch.where(update.unsqueeze(-1), proposed, target)
            self._move_robot(target, self.cfg.transition_steps)
            self._simulate(settle_steps, target)
            if bool(terminal.any()):
                label = f"pivot_{round(100 * waypoint):03d}_terminal"
                self.candidate_diagnostics[f"{label}_ik_rate"] = float(waypoint_valid[terminal].float().mean())
                self.candidate_diagnostics[f"{label}_reached_rate"] = float(reached_waypoint[terminal].float().mean())
        if bool(terminal.any()):
            from ..mdp.geometry import vertical_alignment
            from ..mdp.terms import bilateral_contact

            self.candidate_diagnostics["pivot_terminal_ik_rate"] = float(reached_waypoint[terminal].float().mean())
            self.candidate_diagnostics["pivot_terminal_alignment_mean"] = float(
                vertical_alignment(_tensor(self.vial.data.root_quat_w))[terminal].mean()
            )
            self.candidate_diagnostics["pivot_terminal_bilateral_rate"] = float(
                bilateral_contact(self.env)[terminal].float().mean()
            )
        # A failed stochastic solve holds the last valid joint target and is
        # retried at the next waypoint. Only rows that never found a valid
        # loaded command are rejected here; final contact, lift, alignment,
        # and stability remain mandatory in ``_valid``.
        self.candidate_ik_valid.copy_(source_valid & (~move_mask | reached_waypoint))
        return target

    def _solve_vial_pose(
        self,
        desired_vial_pose: torch.Tensor,
        seed_joint_position: torch.Tensor,
        *,
        reference_vial_pose_w: torch.Tensor | None = None,
        position_tolerance: float | None = None,
        max_joint_delta: float | None = None,
    ) -> torch.Tensor:
        """Solve a grasp-center pose that places the vial at a desired pose.

        ``reference_vial_pose_w`` describes the vial-to-gripper transform to
        preserve.  During approach generation it is a virtual vial centered
        between the open jaws; after grasping it defaults to the live, dynamic
        vial pose.  The object itself is never moved while the robot approaches
        the real tabletop start.
        """
        import warp as wp
        from isaaclab.utils.math import combine_frame_transforms, quat_apply, subtract_frame_transforms

        if reference_vial_pose_w is None:
            vial_position = _tensor(self.vial.data.root_pos_w)
            vial_quaternion = _tensor(self.vial.data.root_quat_w)
        else:
            vial_position = reference_vial_pose_w[:, :3]
            vial_quaternion = reference_vial_pose_w[:, 3:7]
        gripper_position = _tensor(self.robot.data.body_pos_w)[:, self.gripper_body_id].squeeze(1)
        tcp_quaternion = _tensor(self.robot.data.body_quat_w)[:, self.gripper_body_id].squeeze(1)
        tcp_position = gripper_position + quat_apply(
            tcp_quaternion,
            self.grasp_tcp_offset.expand(self.num_envs, -1),
        )
        vial_to_tcp_position, vial_to_tcp_quaternion = subtract_frame_transforms(
            vial_position,
            vial_quaternion,
            tcp_position,
            tcp_quaternion,
        )
        target_position, target_quaternion = combine_frame_transforms(
            desired_vial_pose[:, :3],
            desired_vial_pose[:, 3:7],
            vial_to_tcp_position,
            vial_to_tcp_quaternion,
        )
        self.ik_pose_objective.position_objective.set_target_positions(
            wp.from_torch(target_position.contiguous(), dtype=wp.vec3)
        )
        self.ik_pose_objective.rotation_objective.set_target_rotations(
            wp.from_torch(target_quaternion.contiguous(), dtype=wp.vec4)
        )
        seed = self.ik_joint_seed.clone()
        seed[:, self.ik_coordinate_ids] = seed_joint_position
        self.ik_solver.solve(wp.from_torch(seed.contiguous(), dtype=wp.float32))

        seeds = self.cfg.ik_seeds
        joint_q = wp.to_torch(self.ik_solver.joint_q).reshape(self.num_envs, seeds, -1)
        costs = wp.to_torch(self.ik_solver.costs).reshape(self.num_envs, seeds)
        body_pose = wp.to_torch(self.ik_solver.solver.body_q).reshape(self.num_envs, seeds, -1, 7)
        actual = body_pose[:, :, self.ik_gripper_body_id]
        tcp_offset_position = self.grasp_tcp_offset.expand(self.num_envs, seeds, -1)
        identity_quaternion = torch.zeros_like(actual[:, :, 3:7])
        identity_quaternion[:, :, 3] = 1.0
        actual_tcp_position, actual_tcp_quaternion = combine_frame_transforms(
            actual[:, :, :3].reshape(-1, 3),
            actual[:, :, 3:7].reshape(-1, 4),
            tcp_offset_position.reshape(-1, 3),
            identity_quaternion.reshape(-1, 4),
        )
        actual_tcp_position = actual_tcp_position.reshape(self.num_envs, seeds, 3)
        actual_tcp_quaternion = actual_tcp_quaternion.reshape(self.num_envs, seeds, 4)
        zero_position = torch.zeros_like(vial_to_tcp_position)
        identity_quaternion = torch.zeros_like(vial_to_tcp_quaternion)
        identity_quaternion[:, 3] = 1.0
        tcp_to_vial_position, tcp_to_vial_quaternion = subtract_frame_transforms(
            vial_to_tcp_position,
            vial_to_tcp_quaternion,
            zero_position,
            identity_quaternion,
        )
        predicted_position, predicted_quaternion = combine_frame_transforms(
            actual_tcp_position.reshape(-1, 3),
            actual_tcp_quaternion.reshape(-1, 4),
            tcp_to_vial_position[:, None, :].expand(-1, seeds, -1).reshape(-1, 3),
            tcp_to_vial_quaternion[:, None, :].expand(-1, seeds, -1).reshape(-1, 4),
        )
        predicted_position = predicted_position.reshape(self.num_envs, seeds, 3)
        predicted_quaternion = predicted_quaternion.reshape(self.num_envs, seeds, 4)
        position_error = torch.linalg.vector_norm(predicted_position - desired_vial_pose[:, None, :3], dim=-1)
        tcp_position_error = torch.linalg.vector_norm(actual_tcp_position - target_position[:, None, :], dim=-1)
        local_axis = predicted_position.new_tensor((0.0, 0.0, 1.0)).expand_as(predicted_position)
        predicted_axis = quat_apply(predicted_quaternion.reshape(-1, 4), local_axis.reshape(-1, 3))
        predicted_axis = predicted_axis.reshape(self.num_envs, seeds, 3)
        desired_axis = quat_apply(desired_vial_pose[:, 3:7], local_axis[:, 0])
        axis_alignment = (predicted_axis * desired_axis[:, None]).sum(-1)
        arm_q = joint_q[:, :, self.ik_coordinate_ids[:5]]
        arm_limits = self.ik_joint_limits[:5]
        margin = torch.minimum(arm_q - arm_limits[:, 0], arm_limits[:, 1] - arm_q).amin(-1)
        continuity = torch.linalg.vector_norm(
            arm_q - seed_joint_position[:, None, :5],
            dim=-1,
        )
        if position_tolerance is None:
            position_tolerance = self.cfg.ik_position_tolerance
        seed_valid = (
            torch.isfinite(joint_q).all(-1)
            & torch.isfinite(costs)
            & (position_error <= position_tolerance)
            & (axis_alignment >= math.cos(self.cfg.ik_rotation_tolerance))
            & (margin >= self.cfg.ik_joint_margin)
        )
        if max_joint_delta is not None:
            seed_valid &= continuity <= max_joint_delta
        self.candidate_diagnostics.update(
            {
                "ik_valid_rate": float(seed_valid.any(-1).float().mean()),
                "ik_best_position_error_mean_m": float(position_error.amin(-1).mean()),
                "ik_best_position_error_max_m": float(position_error.amin(-1).max()),
                "ik_best_tcp_error_mean_m": float(tcp_position_error.amin(-1).mean()),
                "ik_best_axis_alignment_mean": float(axis_alignment.amax(-1).mean()),
                "ik_best_axis_alignment_min": float(axis_alignment.amax(-1).min()),
                "ik_best_margin_mean_rad": float(margin.amax(-1).mean()),
                "ik_best_margin_max_rad": float(margin.amax(-1).max()),
                "ik_nearest_joint_delta_mean_rad": float(continuity.amin(-1).mean()),
            }
        )
        # Loaded-object paths must remain on the current kinematic branch.
        # Selecting a distant lower-cost IK solution makes joint interpolation
        # perform a wrist flip even when both endpoint poses are correct.
        valid_cost = continuity.masked_fill(~seed_valid, torch.inf)
        valid = seed_valid.any(-1)
        fallback = torch.nan_to_num(costs, nan=torch.inf, posinf=torch.inf, neginf=torch.inf).argmin(-1)
        selected = torch.where(valid, valid_cost.argmin(-1), fallback)
        rows = torch.arange(self.num_envs, device=self.device)
        solved = joint_q[rows, selected][:, self.ik_coordinate_ids].clone()
        solved[:, -1] = self.gripper_closed_position
        # Rejected IK rows still share the live simulator batch. Keep them at
        # their finite seed pose so one invalid fallback cannot destabilize
        # otherwise independent candidates or poison the next sample batch.
        solved[~valid] = seed_joint_position[~valid]
        self.candidate_ik_valid &= valid
        return solved

    def _transport_vial(
        self,
        target: torch.Tensor,
        desired_pose: torch.Tensor,
        progress: torch.Tensor,
    ) -> torch.Tensor:
        """Raise, then translate an upright vial without a joint-space corner cut."""
        start_pose = torch.cat(
            (
                _tensor(self.vial.data.root_pos_w) - self.env.scene.env_origins,
                _tensor(self.vial.data.root_quat_w),
            ),
            dim=-1,
        ).clone()
        # Preserve the measured orientation. Imposing a fresh full quaternion
        # on a five-DoF arm can rotate the gripper between otherwise reachable
        # endpoints, even though vial yaw has no task meaning.
        desired_pose[:, 3:7] = start_pose[:, 3:7]
        settle_steps = max(1, self.cfg.contact_settle_steps // 6)

        for waypoint in (0.5, 1.0):
            raised_pose = start_pose.clone()
            raised_pose[:, 2] = torch.lerp(start_pose[:, 2], desired_pose[:, 2], waypoint)
            target = self._solve_vial_pose(raised_pose, target, position_tolerance=0.015)
            self._move_robot(target, self.cfg.transition_steps)
            self._simulate(settle_steps, target)

        self._track_safety = True
        self._track_clearance = True
        for waypoint in (0.25, 0.50, 0.75, 1.0):
            waypoint_progress = torch.minimum(progress, torch.full_like(progress, waypoint))
            translated_pose = desired_pose.clone()
            translated_pose[:, :2] = torch.lerp(
                start_pose[:, :2],
                desired_pose[:, :2],
                waypoint_progress.unsqueeze(-1),
            )
            target = self._solve_vial_pose(translated_pose, target, position_tolerance=0.015)
            self._move_robot(target, self.cfg.transition_steps)
            self._simulate(settle_steps, target)
        return target

    def _translate_grasped_vial(
        self,
        target: torch.Tensor,
        translation: torch.Tensor,
        progress: torch.Tensor,
        *,
        move_mask: torch.Tensor | None = None,
        stop_if_separated: bool = False,
        stop_when_upright: bool = False,
    ) -> torch.Tensor:
        """Translate a loaded grasp through measured Cartesian waypoints."""
        from ..mdp.geometry import vertical_alignment
        from ..mdp.terms import grasp_center_w, vial_grasp_point_w

        if move_mask is None:
            move_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        start_pose = torch.cat(
            (
                _tensor(self.vial.data.root_pos_w) - self.env.scene.env_origins,
                _tensor(self.vial.data.root_quat_w),
            ),
            dim=-1,
        ).clone()
        settle_steps = max(1, self.cfg.contact_settle_steps // 6)
        connected_valid = self.candidate_ik_valid.clone()
        active = move_mask.clone()
        for waypoint in (0.25, 0.50, 0.75, 1.0):
            waypoint_progress = torch.minimum(progress, torch.full_like(progress, waypoint))
            desired_pose = start_pose.clone()
            desired_pose[:, :3] += translation * waypoint_progress.unsqueeze(-1)
            # Re-solve against the live, loaded transform at every waypoint.
            # This is ordinary closed-loop position control through the
            # identified USD drives; the vial is never written or constrained.
            self.candidate_ik_valid.fill_(True)
            proposed = self._solve_vial_pose(desired_pose, target, max_joint_delta=0.45)
            waypoint_valid = connected_valid & (~active | self.candidate_ik_valid)
            update = waypoint_valid & active
            target = torch.where(update.unsqueeze(-1), proposed, target)
            self._move_robot(target, self.cfg.transition_steps)
            self._simulate(settle_steps, target)
            centered = torch.linalg.vector_norm(grasp_center_w(self.env) - vial_grasp_point_w(self.env), dim=-1) < 0.05
            # Stop advancing a row as soon as the load leaves the jaws. This
            # prevents the Cartesian solver from chasing a dropped vial across
            # the table; the measured terminal state may still be regrasped if
            # the release happened only after the vial reached upright.
            if stop_if_separated:
                active &= centered
            if stop_when_upright:
                active &= vertical_alignment(_tensor(self.vial.data.root_quat_w)) < 0.88
        self.candidate_ik_valid.copy_(connected_valid)
        return target

    def _solve_pregrasp(
        self,
        vial_pose: torch.Tensor,
        seed_joint_position: torch.Tensor,
        *,
        ik_seed_joint_position: torch.Tensor | None = None,
        vertical_offset: float = 0.0,
        target_position_bias: torch.Tensor | None = None,
        target_quaternion: torch.Tensor | None = None,
        randomize_solution: bool = False,
    ) -> torch.Tensor:
        """Solve an open-jaw pose centered on the vial's physical grasp point."""
        import warp as wp
        from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_from_matrix

        from ..mdp.terms import VIAL_GRASP_OFFSET

        offset = vial_pose.new_tensor(VIAL_GRASP_OFFSET).expand(self.num_envs, -1)
        target_position = vial_pose[:, :3] + quat_apply(vial_pose[:, 3:7], offset)
        target_position[:, 2] += vertical_offset
        if target_position_bias is not None:
            target_position += target_position_bias
        self._last_pregrasp_target_position = target_position.clone()
        vial_axis = quat_apply(
            vial_pose[:, 3:7],
            vial_pose.new_tensor((0.0, 0.0, 1.0)).expand(self.num_envs, -1),
        )
        if target_quaternion is None:
            world_up = vial_pose.new_tensor((0.0, 0.0, 1.0)).expand_as(vial_axis)
            # Choose the same cap-first wrist branch used in the workshop
            # demonstrations.  The opposite, cylinder-symmetric solution is
            # kinematically valid but starts with the wrist rolled by roughly
            # pi radians and cannot continue through the supported pickup.
            pad_sign = torch.ones(self.num_envs, device=self.device)
            pad_axis = vial_axis * pad_sign.unsqueeze(-1)
            closing_axis = torch.linalg.cross(pad_axis, world_up, dim=-1)
            closing_norm = torch.linalg.vector_norm(closing_axis, dim=-1, keepdim=True)
            # For an upright vial, vial axis and world-up are parallel. Use a
            # horizontal closing direction instead of normalizing a zero
            # cross-product; pad +Y remains aligned with the vial.
            fallback_closing = vial_pose.new_tensor((1.0, 0.0, 0.0)).expand_as(closing_axis)
            closing_axis = torch.where(
                closing_norm > 1.0e-4,
                closing_axis / closing_norm.clamp_min(1.0e-6),
                fallback_closing,
            )
            gripper_up = torch.linalg.cross(closing_axis, pad_axis, dim=-1)
            # Rotation-matrix columns are gripper +X (closing), +Y (the
            # fingertip pad's long direction), and +Z (up/approach).
            target_quaternion = quat_from_matrix(torch.stack((closing_axis, pad_axis, gripper_up), dim=-1))
        self.ik_pose_objective.position_objective.set_target_positions(
            wp.from_torch(target_position.contiguous(), dtype=wp.vec3)
        )
        self.ik_pose_objective.rotation_objective.set_target_rotations(
            wp.from_torch(target_quaternion.contiguous(), dtype=wp.vec4)
        )
        if ik_seed_joint_position is None:
            ik_seed_joint_position = seed_joint_position
        seed = self.ik_joint_seed.clone()
        seed[:, self.ik_coordinate_ids] = ik_seed_joint_position
        self.ik_solver.solve(wp.from_torch(seed.contiguous(), dtype=wp.float32))

        seeds = self.cfg.ik_seeds
        joint_q = wp.to_torch(self.ik_solver.joint_q).reshape(self.num_envs, seeds, -1)
        costs = wp.to_torch(self.ik_solver.costs).reshape(self.num_envs, seeds)
        body_pose = wp.to_torch(self.ik_solver.solver.body_q).reshape(self.num_envs, seeds, -1, 7)
        gripper_pose = body_pose[:, :, self.ik_gripper_body_id]
        tcp_offset = self.grasp_tcp_offset.expand(self.num_envs, seeds, -1)
        identity = torch.zeros_like(gripper_pose[:, :, 3:7])
        identity[:, :, 3] = 1.0
        actual_tcp, _ = combine_frame_transforms(
            gripper_pose[:, :, :3].reshape(-1, 3),
            gripper_pose[:, :, 3:7].reshape(-1, 4),
            tcp_offset.reshape(-1, 3),
            identity.reshape(-1, 4),
        )
        actual_tcp = actual_tcp.reshape(self.num_envs, seeds, 3)
        position_error = torch.linalg.vector_norm(actual_tcp - target_position[:, None], dim=-1)

        # The SO-101 pads close along gripper +X. A cylinder grasp is viable
        # when that direction is perpendicular to the vial's long axis; spin
        # around the cylinder is intentionally left free.
        closing_axis = gripper_pose.new_tensor((1.0, 0.0, 0.0)).expand(self.num_envs, seeds, -1)
        closing_axis = quat_apply(gripper_pose[:, :, 3:7].reshape(-1, 4), closing_axis.reshape(-1, 3))
        closing_axis = closing_axis.reshape(self.num_envs, seeds, 3)
        closing_alignment = (closing_axis * vial_axis[:, None]).sum(-1).abs()
        pad_axis = gripper_pose.new_tensor((0.0, 1.0, 0.0)).expand(self.num_envs, seeds, -1)
        pad_axis = quat_apply(gripper_pose[:, :, 3:7].reshape(-1, 4), pad_axis.reshape(-1, 3))
        pad_axis = pad_axis.reshape(self.num_envs, seeds, 3)
        pad_alignment = (pad_axis * vial_axis[:, None]).sum(-1).abs()

        arm_q = joint_q[:, :, self.ik_coordinate_ids[:5]]
        arm_limits = self.ik_joint_limits[:5]
        margin = torch.minimum(arm_q - arm_limits[:, 0], arm_limits[:, 1] - arm_q).amin(-1)
        seed_valid = (
            torch.isfinite(joint_q).all(-1)
            & torch.isfinite(costs)
            & (position_error <= self.cfg.ik_position_tolerance)
            & (closing_alignment <= math.sin(self.cfg.ik_rotation_tolerance))
            & (pad_alignment >= self.cfg.min_grasp_pad_alignment)
            & (margin >= self.cfg.ik_joint_margin)
        )
        continuity = torch.linalg.vector_norm(
            arm_q - ik_seed_joint_position[:, None, :5],
            dim=-1,
        )
        score = closing_alignment + (1.0 - pad_alignment) + 0.05 * continuity + 10.0 * position_error
        valid_score = score.masked_fill(~seed_valid, torch.inf)
        valid = seed_valid.any(-1)
        fallback = torch.nan_to_num(score, nan=torch.inf, posinf=torch.inf, neginf=torch.inf).argmin(-1)
        if randomize_solution:
            weights = seed_valid.float()
            weights[~valid] = 1.0
            sampled = torch.multinomial(weights, 1, replacement=True, generator=self.random).squeeze(-1)
            selected = torch.where(valid, sampled, fallback)
        else:
            selected = torch.where(valid, valid_score.argmin(-1), fallback)
        rows = torch.arange(self.num_envs, device=self.device)
        solved = joint_q[rows, selected][:, self.ik_coordinate_ids].clone()
        solved[:, -1] = PREGRASP_GRIPPER_POSITION
        solved[~valid] = seed_joint_position[~valid]
        solved[~valid, -1] = PREGRASP_GRIPPER_POSITION
        self._last_pregrasp_quaternion = gripper_pose[rows, selected, 3:7].clone()
        self.candidate_ik_valid &= valid
        for name, value in zip(
            ("pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper"),
            solved.mean(0),
            strict=True,
        ):
            self.candidate_diagnostics[f"pregrasp_{name}_command_mean_rad"] = float(value)
        self.candidate_diagnostics.update(
            {
                "pregrasp_ik_valid_rate": float(valid.float().mean()),
                "pregrasp_finite_rate": float(
                    (torch.isfinite(joint_q).all(-1) & torch.isfinite(costs)).any(-1).float().mean()
                ),
                "pregrasp_position_rate": float(
                    (position_error <= self.cfg.ik_position_tolerance).any(-1).float().mean()
                ),
                "pregrasp_closing_rate": float(
                    (closing_alignment <= math.sin(self.cfg.ik_rotation_tolerance)).any(-1).float().mean()
                ),
                "pregrasp_pad_rate": float((pad_alignment >= self.cfg.min_grasp_pad_alignment).any(-1).float().mean()),
                "pregrasp_margin_rate": float((margin >= self.cfg.ik_joint_margin).any(-1).float().mean()),
                "pregrasp_position_error_mean_m": float(position_error.amin(-1).mean()),
                "pregrasp_closing_alignment_mean": float(closing_alignment.amin(-1).mean()),
                "pregrasp_pad_alignment_mean": float(pad_alignment.amax(-1).mean()),
                "pregrasp_best_margin_mean_rad": float(margin.amax(-1).mean()),
                "pregrasp_best_margin_max_rad": float(margin.amax(-1).max()),
            }
        )
        return solved

    def _center_open_gripper(
        self,
        target: torch.Tensor,
        target_quaternion: torch.Tensor,
        *,
        corrections: int = 1,
    ) -> torch.Tensor:
        """Use bounded Cartesian feedback to center the dynamic open jaws.

        Finite-stiffness position drives have pose-dependent steady-state
        error under gravity. This controller changes only normal joint
        position commands; every correction is simulated with the original
        USD actuator and the vial remains fully dynamic.
        """
        from isaaclab.utils.math import quat_apply

        from ..mdp.terms import vial_grasp_point_w

        bias = torch.zeros((self.num_envs, 3), device=self.device)
        for _ in range(corrections):
            gripper_position = _tensor(self.robot.data.body_pos_w)[:, self.gripper_body_id].squeeze(1)
            gripper_quaternion = _tensor(self.robot.data.body_quat_w)[:, self.gripper_body_id].squeeze(1)
            live_tcp = gripper_position + quat_apply(
                gripper_quaternion,
                self.grasp_tcp_offset.expand(self.num_envs, -1),
            )
            error = live_tcp - vial_grasp_point_w(self.env)
            bias -= error
            bias_norm = torch.linalg.vector_norm(bias, dim=-1, keepdim=True)
            bias *= (0.06 / bias_norm.clamp_min(0.06)).clamp_max(1.0)
            live_pose = torch.cat(
                (
                    _tensor(self.vial.data.root_pos_w) - self.env.scene.env_origins,
                    _tensor(self.vial.data.root_quat_w),
                ),
                dim=-1,
            ).clone()
            target = self._solve_pregrasp(
                live_pose,
                target,
                target_position_bias=bias,
                target_quaternion=target_quaternion,
            )
            self._move_robot(target, max(1, self.cfg.transition_steps // 2))
            self._simulate(max(1, self.cfg.contact_settle_steps // 3), target)
        return target

    def _track_loaded_pose(
        self,
        target: torch.Tensor,
        desired_pose: torch.Tensor,
        *,
        move_mask: torch.Tensor | None = None,
        corrections: int = 3,
    ) -> torch.Tensor:
        """Track one loaded Cartesian pose with bounded servo feedback.

        The first IK solution establishes reachability. Later solutions are
        optional feedback corrections for finite drive deflection: a row keeps
        its last reachable target when a correction would cross an IK margin,
        and the final physics checks decide whether the motion succeeded.
        """
        if move_mask is None:
            move_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        required_valid = self.candidate_ik_valid.clone()
        for correction in range(corrections):
            self.candidate_ik_valid.fill_(True)
            proposed = self._solve_vial_pose(desired_pose, target)
            correction_valid = self.candidate_ik_valid.clone()
            if correction == 0:
                required_valid &= ~move_mask | correction_valid
            update = move_mask & required_valid & correction_valid
            target = torch.where(update.unsqueeze(-1), proposed, target)
            self._move_robot(target, max(1, self.cfg.transition_steps // corrections))
            self._simulate(max(1, self.cfg.contact_settle_steps // corrections), target)
        self.candidate_ik_valid.copy_(required_valid)
        return target

    def _correct_upright_endpoint(self, target: torch.Tensor, move_mask: torch.Tensor) -> torch.Tensor:
        """Close the residual reorientation error through ordinary joint targets."""
        from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul

        live_pose = torch.cat(
            (
                _tensor(self.vial.data.root_pos_w) - self.env.scene.env_origins,
                _tensor(self.vial.data.root_quat_w),
            ),
            dim=-1,
        ).clone()
        local_axis = live_pose.new_tensor((0.0, 0.0, 1.0)).expand(self.num_envs, -1)
        vial_axis = quat_apply(live_pose[:, 3:7], local_axis)
        world_up = live_pose.new_tensor((0.0, 0.0, 1.0)).expand_as(vial_axis)
        correction_axis = torch.linalg.cross(vial_axis, world_up, dim=-1)
        correction_norm = torch.linalg.vector_norm(correction_axis, dim=-1, keepdim=True)
        fallback_axis = live_pose.new_tensor((1.0, 0.0, 0.0)).expand_as(correction_axis)
        correction_axis = torch.where(
            correction_norm > 1.0e-6,
            correction_axis / correction_norm.clamp_min(1.0e-6),
            fallback_axis,
        )
        correction_angle = torch.acos((vial_axis * world_up).sum(-1).clamp(-1.0, 1.0))
        correction = quat_from_angle_axis(correction_angle, correction_axis)
        desired_pose = live_pose.clone()
        desired_pose[:, 3:7] = quat_mul(correction, live_pose[:, 3:7])
        self._target_vial_quaternion.copy_(
            torch.where(move_mask.unsqueeze(-1), desired_pose[:, 3:7], self._target_vial_quaternion)
        )
        return self._track_loaded_pose(target, desired_pose, move_mask=move_mask, corrections=4)

    def _regrasp_upright_vial(self, target: torch.Tensor, terminal: torch.Tensor) -> torch.Tensor:
        """Open and physically regrasp terminal upright pivot candidates."""
        from ..mdp.geometry import vertical_alignment
        from ..mdp.terms import bilateral_contact, grasp_center_w, vial_grasp_point_w

        alignment = vertical_alignment(_tensor(self.vial.data.root_quat_w))
        speed = torch.linalg.vector_norm(_tensor(self.vial.data.root_lin_vel_w), dim=-1)
        eligible = terminal & (alignment > 0.88) & (speed < 0.15)

        open_target = target.clone()
        open_target[eligible, -1] = PREGRASP_GRIPPER_POSITION
        self._move_robot(open_target, self.cfg.grasp_close_steps)
        self._simulate(self.cfg.contact_settle_steps, open_target)

        alignment = vertical_alignment(_tensor(self.vial.data.root_quat_w))
        speed = torch.linalg.vector_norm(_tensor(self.vial.data.root_lin_vel_w), dim=-1)
        eligible &= (alignment > 0.88) & (speed < 0.08)
        vial_pose = torch.cat(
            (
                _tensor(self.vial.data.root_pos_w) - self.env.scene.env_origins,
                _tensor(self.vial.data.root_quat_w),
            ),
            dim=-1,
        ).clone()

        self.candidate_ik_valid.fill_(True)
        overhead = self._solve_pregrasp(vial_pose, open_target, vertical_offset=0.025)
        overhead_valid = self.candidate_ik_valid.clone()
        overhead_target = torch.where((eligible & overhead_valid).unsqueeze(-1), overhead, open_target)
        self._move_robot(overhead_target, self.cfg.transition_steps)
        self._simulate(max(1, self.cfg.contact_settle_steps // 3), overhead_target)

        self.candidate_ik_valid.fill_(True)
        pregrasp = self._solve_pregrasp(
            vial_pose,
            overhead_target,
            target_quaternion=self._last_pregrasp_quaternion,
        )
        approach_valid = eligible & overhead_valid & self.candidate_ik_valid
        pregrasp_target = torch.where(approach_valid.unsqueeze(-1), pregrasp, overhead_target)
        self._move_robot(pregrasp_target, self.cfg.transition_steps)
        self._simulate(max(1, self.cfg.contact_settle_steps // 3), pregrasp_target)

        close_target = pregrasp_target.clone()
        close_target[approach_valid, -1] = self.gripper_closed_position
        self._move_robot(close_target, self.cfg.grasp_close_steps)
        self._simulate(self.cfg.contact_settle_steps, close_target)
        centered = torch.linalg.vector_norm(grasp_center_w(self.env) - vial_grasp_point_w(self.env), dim=-1) < 0.030
        success = approach_valid & bilateral_contact(self.env) & centered
        success &= vertical_alignment(_tensor(self.vial.data.root_quat_w)) > 0.88
        self.candidate_ik_valid.copy_(~terminal | success)
        self.candidate_diagnostics["upright_before_regrasp_rate"] = float(eligible[terminal].float().mean())
        self.candidate_diagnostics["upright_regrasp_rate"] = float(success[terminal].float().mean())
        return close_target

    def _simulate(self, steps: int, target: torch.Tensor) -> None:
        for _ in range(steps):
            self.robot.set_joint_position_target_index(target=target)
            self.robot.set_joint_velocity_target_index(target=self._zeros)
            self.env.scene.write_data_to_sim()
            self.env.sim.step()
            self.env.scene.update(self.env.physics_dt)
            if self._track_safety:
                from ..mdp.terms import rack_clearance_violation, undesired_rack_contact, unsafe_rack_contact

                # Endpoint validation is insufficient: a candidate can hit a
                # rail, rebound, and look calm after settling. Preserve every
                # violation along the transport/insertion rollout.
                if self._track_clearance:
                    self.candidate_clearance_violation |= rack_clearance_violation(self.env).bool()
                self.candidate_rack_contact |= undesired_rack_contact(self.env) > 0.0
                self.candidate_unsafe |= self.candidate_clearance_violation | unsafe_rack_contact(self.env)

    def _record_grasp_diagnostics(self, prefix: str) -> None:
        """Record finite measurements for IK-valid grasp candidates only."""
        from isaaclab.utils.math import quat_apply

        from ..mdp.terms import contact_state, fingertip_positions_w, grasp_center_w, vial_grasp_point_w

        fixed, moving = fingertip_positions_w(self.env)
        vial_point = vial_grasp_point_w(self.env)
        center = grasp_center_w(self.env)
        gripper_quaternion = _tensor(self.robot.data.body_quat_w)[:, self.gripper_body_id].squeeze(1)
        closing_axis = quat_apply(
            gripper_quaternion,
            gripper_quaternion.new_tensor((1.0, 0.0, 0.0)).expand(self.num_envs, -1),
        )
        mask = self.candidate_ik_valid.clone()
        mask &= torch.isfinite(fixed).all(-1) & torch.isfinite(moving).all(-1)
        mask &= torch.isfinite(vial_point).all(-1) & torch.isfinite(center).all(-1)
        if not bool(mask.any()):
            return
        contacts = contact_state(self.env)[mask]
        fixed_delta = vial_point[mask] - fixed[mask]
        moving_delta = vial_point[mask] - moving[mask]
        self.candidate_diagnostics.update(
            {
                f"{prefix}_valid_rows": float(mask.sum()),
                f"{prefix}_bilateral_rate": float(contacts[:, 2].mean()),
                f"{prefix}_fixed_contact_rate": float(contacts[:, 0].mean()),
                f"{prefix}_moving_contact_rate": float(contacts[:, 1].mean()),
                f"{prefix}_center_error_mean_m": float(
                    torch.linalg.vector_norm(center[mask] - vial_point[mask], dim=-1).mean()
                ),
                f"{prefix}_fixed_distance_mean_m": float(torch.linalg.vector_norm(fixed_delta, dim=-1).mean()),
                f"{prefix}_moving_distance_mean_m": float(torch.linalg.vector_norm(moving_delta, dim=-1).mean()),
                f"{prefix}_fixed_closing_offset_mean_m": float((fixed_delta * closing_axis[mask]).sum(-1).mean()),
                f"{prefix}_moving_closing_offset_mean_m": float((moving_delta * closing_axis[mask]).sum(-1).mean()),
            }
        )

    def _write_robot(self, joint_position: torch.Tensor) -> None:
        self.robot.write_joint_position_to_sim_index(position=joint_position)
        self.robot.write_joint_velocity_to_sim_index(velocity=self._zeros)
        self.robot.set_joint_position_target_index(target=joint_position)
        self.action_term._joint_target.copy_(joint_position)

    def _write_vial(self, local_pose: torch.Tensor) -> None:
        world_pose = local_pose.clone()
        world_pose[:, :3] += self.env.scene.env_origins
        self.vial.write_root_pose_to_sim_index(root_pose=world_pose)
        self.vial.write_root_velocity_to_sim_index(root_velocity=self._zeros)

    def _commit_reset_state(self) -> None:
        """Commit explicit state writes before advancing Newton physics.

        Rejection sampling deliberately visits failed candidates. Clearing the
        scene's cached asset/sensor state and evaluating forward kinematics
        prevents a rejected world's stale body or contact buffers from being
        observed as the next finite candidate.
        """
        self.env.scene.reset()
        self.env.scene.write_data_to_sim()
        self.env.sim.forward()
        self.env.scene.update(self.env.physics_dt)

    def _append_seed_bank(self, name: str, valid: torch.Tensor, target: torch.Tensor) -> None:
        """Cache dynamically reached states for a later connected branch."""
        if not bool(valid.any()):
            return
        part = {
            "joint_position": _tensor(self.robot.data.joint_pos)[valid].detach().clone(),
            "joint_target": target[valid].detach().clone(),
            "vial_pose": torch.cat(
                (
                    _tensor(self.vial.data.root_pos_w)[valid] - self.env.scene.env_origins[valid],
                    _tensor(self.vial.data.root_quat_w)[valid],
                ),
                dim=-1,
            ).detach(),
        }
        if name in self._seed_banks:
            part = {key: torch.cat((self._seed_banks[name][key], value), dim=0) for key, value in part.items()}
        maximum = self.cfg.batch_size * 4
        self._seed_banks[name] = {key: value[-maximum:] for key, value in part.items()}

    def _restore_seed_bank(self, name: str, *, settle_steps: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Restore a physics-validated state across the generator batch."""
        bank = self._seed_banks[name]
        rows = torch.randint(
            bank["joint_position"].shape[0],
            (self.num_envs,),
            device=self.device,
            generator=self.random,
        )
        joint_position = bank["joint_position"][rows]
        target = bank["joint_target"][rows].clone()
        local_pose = bank["vial_pose"][rows]
        self._write_robot(joint_position)
        self._write_vial(local_pose)
        self.robot.set_joint_position_target_index(target=target)
        self.action_term._joint_target.copy_(target)
        self._commit_reset_state()
        if settle_steps is None:
            settle_steps = max(1, self.cfg.contact_settle_steps // 3)
        self._simulate(settle_steps, target)
        self._target_vial_quaternion.copy_(local_pose[:, 3:7])
        initial_world_position = local_pose[:, :3] + self.env.scene.env_origins
        return target, initial_world_position

    def _move_robot(self, target: torch.Tensor, steps: int) -> None:
        """Move to a joint waypoint smoothly while every object remains dynamic."""
        start = _tensor(self.robot.data.joint_pos).clone()
        for step in range(1, steps + 1):
            blend = step / steps
            self._simulate(1, torch.lerp(start, target, blend))

    def _follow_workshop_segment(
        self,
        target: torch.Tensor,
        phase: int,
        progress: torch.Tensor,
    ) -> torch.Tensor:
        """Replay one sparse real-robot segment through the Sys-ID drives.

        Only robot position targets are replayed. The vial remains a free
        rigid body, so later rejection checks—not the recording—determine its
        pose, contacts, stability, and clearance.
        """
        waypoints = target.new_tensor(WORKSHOP_TASK_WAYPOINTS[phase])
        limits = self.ik_joint_limits
        if not bool(((waypoints >= limits[:, 0]) & (waypoints <= limits[:, 1])).all()):
            raise RuntimeError(f"Workshop phase {phase} contains a command outside the USD soft limits.")

        segment_start = target.clone()
        count = waypoints.shape[0]
        for index, waypoint in enumerate(waypoints):
            start_fraction = index / count
            local_progress = ((progress - start_fraction) * count).clamp(0.0, 1.0)
            base = segment_start if index == 0 else waypoints[index - 1].expand_as(target)
            proposed = torch.lerp(base, waypoint.expand_as(target), local_progress.unsqueeze(-1))
            active = progress > start_fraction
            target = torch.where(active.unsqueeze(-1), proposed, target)
            self._move_robot(target, self.cfg.transition_steps)
            self._simulate(max(1, self.cfg.contact_settle_steps // 6), target)
        return target

    def _settle_configuration(self, desired: torch.Tensor) -> torch.Tensor:
        """Settle a USD-drive command without altering or compensating it."""
        self._simulate(self.cfg.contact_settle_steps, desired)
        measured_error = torch.linalg.vector_norm(_tensor(self.robot.data.joint_pos) - desired, dim=-1)
        mask = self.candidate_ik_valid & torch.isfinite(measured_error)
        if bool(mask.any()):
            self.candidate_diagnostics["settle_joint_error_mean_rad"] = float(measured_error[mask].mean())
            joint_error = (_tensor(self.robot.data.joint_pos) - desired).abs()[mask].mean(0)
            for name, error in zip(
                ("pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper"),
                joint_error,
                strict=True,
            ):
                self.candidate_diagnostics[f"settle_{name}_error_mean_rad"] = float(error)
        return desired

    def _park_vial(self) -> None:
        pose = torch.zeros((self.num_envs, 7), device=self.device)
        pose[:, :3] = self.env.scene.env_origins + pose.new_tensor((0.0, 0.0, 1.0))
        pose[:, 6] = 1.0
        self.vial.write_root_pose_to_sim_index(root_pose=pose)
        self.vial.write_root_velocity_to_sim_index(root_velocity=self._zeros)

    def _reset_rack(self) -> None:
        """Restore the dynamic rack before each independent candidate batch."""
        pose = _tensor(self.rack.data.default_root_pose).clone()
        pose[:, :3] += self.env.scene.env_origins
        self.rack.write_root_pose_to_sim_index(root_pose=pose)
        self.rack.write_root_velocity_to_sim_index(root_velocity=self._zeros)

    def _canonical_home_joint_positions(self) -> torch.Tensor:
        """Return the exact operational pose used to begin every full task."""
        # Phase zero is the public start distribution, not an approach
        # curriculum. Keeping this pose exact prevents canonical evaluation
        # from beginning with the vial already at or between the jaws.
        base = self._zeros.new_tensor(WORKSHOP_INITIAL_JOINT_POSITION).repeat(self.num_envs, 1)
        limits = _tensor(self.robot.data.soft_joint_pos_limits)
        return base.clamp(limits[..., 0], limits[..., 1])

    def _tabletop_pose(self, phase: int) -> torch.Tensor:
        pose = torch.zeros((self.num_envs, 7), device=self.device)
        pose[:, :3] = pose.new_tensor(TABLETOP_VIAL_POSITION)
        half_range = pose.new_tensor(self.cfg.vial_position_half_range)
        pose[:, :2] += torch.empty((self.num_envs, 2), device=self.device).uniform_(
            -1.0, 1.0, generator=self.random
        ) * half_range
        # The workshop randomizes roll about the vial axis, not its tabletop
        # heading. A modest yaw band covers setup error without presenting the
        # cap beyond the small arm's reliable continuation workspace.
        yaw_range = TABLETOP_VIAL_HEADING_RANGE if phase == 0 else (-0.12, 0.12)
        yaw = torch.empty(self.num_envs, device=self.device).uniform_(*yaw_range, generator=self.random)
        half = 0.5 * yaw
        yaw_quat = torch.stack((torch.zeros_like(yaw), torch.zeros_like(yaw), half.sin(), half.cos()), dim=-1)
        from isaaclab.utils.math import quat_mul

        pose[:, 3:7] = quat_mul(yaw_quat, pose.new_tensor(_HORIZONTAL_QUATERNION).expand(self.num_envs, -1))
        return pose

    def _approach_candidate(
        self,
        phase: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move the open gripper toward a stationary tabletop vial.

        Phase 0 is the exact operational home pose, phase 1 spans physical jaw
        closure, and ``None`` reaches the open
        pregrasp endpoint before seeding a load-bearing grasp. All rows
        belong to one connected task trajectory beginning at the same
        distribution used for evaluation.
        """
        start = self._canonical_home_joint_positions()
        start[:, -1] = PREGRASP_GRIPPER_POSITION
        self._park_vial()
        self._write_robot(start)
        self._commit_reset_state()
        self._simulate(self.cfg.articulation_settle_steps, start)

        # First let the dynamic vial find its real resting height and
        # orientation. IK must target this measured pose, not the authored
        # collision-free spawn pose several millimetres above the mat.
        self._write_vial(self._tabletop_pose(0))
        self._commit_reset_state()
        self._simulate(self.cfg.contact_settle_steps, start)
        local_pose = torch.cat(
            (
                _tensor(self.vial.data.root_pos_w) - self.env.scene.env_origins,
                _tensor(self.vial.data.root_quat_w),
            ),
            dim=-1,
        ).clone()
        overhead = self._solve_pregrasp(
            local_pose,
            start,
            ik_seed_joint_position=start.new_tensor(WORKSHOP_PREGRASP_JOINT_POSITION).expand_as(start),
            vertical_offset=0.03,
            randomize_solution=True,
        )
        overhead_valid = self.candidate_ik_valid.clone()
        overhead_diagnostics = {
            key.replace("pregrasp_", "overhead_", 1): value
            for key, value in self.candidate_diagnostics.items()
            if key.startswith("pregrasp_")
        }
        pregrasp = self._solve_pregrasp(
            local_pose,
            overhead,
            target_quaternion=self._last_pregrasp_quaternion,
        )
        self.candidate_diagnostics.update(overhead_diagnostics)
        self.candidate_diagnostics["overhead_ik_valid_rate"] = float(overhead_valid.float().mean())
        self.candidate_diagnostics["connected_ik_valid_rate"] = float(self.candidate_ik_valid.float().mean())
        self._last_pregrasp_joint_position = pregrasp.clone()

        self._write_robot(start)
        if phase is None or phase in (0, 1):
            # Home-to-overhead motion is reset initialization, not part of the
            # manipulation horizon. Keep the vial parked while the arm makes
            # that unconstrained joint-space sweep, then establish the real
            # tabletop state before the validated descent begins.
            self._park_vial()
            self._write_robot(overhead)
            overhead_command = self._settle_configuration(overhead)
            self._write_robot(pregrasp)
            pregrasp_command = self._settle_configuration(pregrasp)
            self._write_robot(overhead)
            self._simulate(self.cfg.contact_settle_steps, overhead_command)
            self._write_vial(local_pose)
            self._simulate(self.cfg.contact_settle_steps, overhead_command)
        if phase in (0, 1):
            # Calibrate the terminal command using the same dynamic,
            # finite-stiffness approach that seeds the grasp phase. Restore
            # the scene afterwards so every serialized candidate is reached
            # through its own physical descent/closure rollout.
            for step in range(1, self.cfg.transition_steps + 1):
                blend = step / self.cfg.transition_steps
                self._simulate(1, torch.lerp(overhead_command, pregrasp_command, blend))
            self._simulate(self.cfg.contact_settle_steps, pregrasp_command)
            pregrasp_command = self._center_open_gripper(
                pregrasp_command,
                self._last_pregrasp_quaternion.clone(),
            )
            self._write_robot(overhead)
            self._simulate(self.cfg.contact_settle_steps, overhead_command)
            self._write_vial(local_pose)
            self._simulate(self.cfg.contact_settle_steps, overhead_command)
        # Measure all later drift from this physically settled start state.
        initial_world_position = _tensor(self.vial.data.root_pos_w).clone()
        if phase == 0:
            # Canonical episodes always begin at home. The policy must execute
            # the complete home-to-overhead-to-pregrasp approach itself.
            fraction = torch.zeros(self.num_envs, device=self.device)
            first_progress = (2.0 * fraction).clamp_max(1.0)
            second_progress = (2.0 * fraction - 1.0).clamp(0.0, 1.0)
            first_target = torch.lerp(start, overhead_command, first_progress.unsqueeze(-1))
            target = torch.lerp(overhead_command, pregrasp_command, second_progress.unsqueeze(-1))
            target = torch.where((fraction < 0.5).unsqueeze(-1), first_target, target)
        else:
            if phase == 1:
                # Phase one is the continuous physical jaw-closure segment.
                # Phase two begins only after an independent load-bearing
                # proof lift, so weak static grasps are never serialized as
                # proven grasp resets.
                closure_fraction = torch.rand(self.num_envs, device=self.device, generator=self.random)
                arm_fraction = torch.ones_like(closure_fraction)
                fraction = (1.0 + closure_fraction) / 3.0
            else:
                fraction = torch.ones(self.num_envs, device=self.device)
                arm_fraction = fraction
            target = torch.lerp(overhead, pregrasp, arm_fraction.unsqueeze(-1))
        if phase is None:
            # A load-bearing grasp seed must be reached through the same
            # collision-aware descent available to the policy.
            for step in range(1, self.cfg.transition_steps + 1):
                blend = step / self.cfg.transition_steps
                self._simulate(1, torch.lerp(overhead_command, pregrasp_command, blend))
            target = pregrasp_command
            self._simulate(self.cfg.contact_settle_steps, target)
            target = self._center_open_gripper(target, self._last_pregrasp_quaternion.clone())
            from isaaclab.utils.math import quat_apply

            from ..mdp.terms import vial_grasp_point_w

            gripper_position = _tensor(self.robot.data.body_pos_w)[:, self.gripper_body_id].squeeze(1)
            gripper_quaternion = _tensor(self.robot.data.body_quat_w)[:, self.gripper_body_id].squeeze(1)
            live_tcp = gripper_position + quat_apply(
                gripper_quaternion,
                self.grasp_tcp_offset.expand(self.num_envs, -1),
            )
            tcp_error_vector = live_tcp - vial_grasp_point_w(self.env)
            tcp_error = torch.linalg.vector_norm(tcp_error_vector, dim=-1)
            vial_drift = torch.linalg.vector_norm(_tensor(self.vial.data.root_pos_w) - initial_world_position, dim=-1)
            mask = self.candidate_ik_valid & torch.isfinite(tcp_error) & torch.isfinite(vial_drift)
            if bool(mask.any()):
                self.candidate_diagnostics["pregrasp_live_tcp_error_mean_m"] = float(tcp_error[mask].mean())
                self.candidate_diagnostics["pregrasp_live_vial_drift_mean_m"] = float(vial_drift[mask].mean())
                for axis, value in zip(("x", "y", "z"), tcp_error_vector[mask].mean(0), strict=True):
                    self.candidate_diagnostics[f"pregrasp_live_tcp_error_{axis}_mean_m"] = float(value)
        elif phase in (0, 1):
            if phase == 0:
                # Restore and settle the exact home start with the vial on the
                # table. A zero fraction intentionally executes no hidden
                # portion of either approach leg.
                self._write_robot(start)
                self._write_vial(local_pose)
                self._commit_reset_state()
                first_progress = (2.0 * fraction).clamp_max(1.0)
                first_target = torch.lerp(start, overhead_command, first_progress.unsqueeze(-1))
                for step in range(1, self.cfg.transition_steps + 1):
                    blend = step / self.cfg.transition_steps
                    self._simulate(1, torch.lerp(start, first_target, blend))
                second_rows = fraction >= 0.5
                if bool(second_rows.any()):
                    second_progress = (2.0 * fraction - 1.0).clamp(0.0, 1.0)
                    second_target = torch.lerp(
                        overhead_command,
                        pregrasp_command,
                        second_progress.unsqueeze(-1),
                    )
                    for step in range(1, self.cfg.transition_steps + 1):
                        blend = step / self.cfg.transition_steps
                        proposed = torch.lerp(overhead_command, second_target, blend)
                        command = torch.where(second_rows.unsqueeze(-1), proposed, first_target)
                        self._simulate(1, command)
                    target = torch.where(second_rows.unsqueeze(-1), second_target, first_target)
                self._simulate(self.cfg.contact_settle_steps, target)
                self.candidate_difficulty.copy_(fraction / 7.0)
                valid = self._valid(phase, initial_world_position)
                return valid, target, local_pose, initial_world_position
            target = torch.lerp(overhead_command, pregrasp_command, arm_fraction.unsqueeze(-1))
            for step in range(1, self.cfg.transition_steps + 1):
                blend = step / self.cfg.transition_steps
                self._simulate(1, torch.lerp(overhead_command, target, blend))
            close_target = target.clone()
            close_target[:, -1] = torch.lerp(
                close_target[:, -1],
                torch.full_like(close_target[:, -1], self.gripper_closed_position),
                closure_fraction,
            )
            self._move_robot(close_target, self.cfg.grasp_close_steps)
            target = close_target
            self._simulate(self.cfg.contact_settle_steps // 3, target)
            self._simulate(self.cfg.contact_settle_steps, target)
        else:
            # Earlier reset phases represent stable points along the approach
            # rather than an open-loop planner rollout.
            self._write_robot(target)
            self._simulate(self.cfg.contact_settle_steps // 3, target)
        if phase is not None:
            self.candidate_difficulty.copy_((phase + fraction).clamp_max(7.0) / 7.0)
        valid = self._valid(0 if phase is None else phase, initial_world_position)
        return valid, target, local_pose, initial_world_position

    def _valid(self, phase: int, initial_vial_position: torch.Tensor) -> torch.Tensor:
        from ..mdp.terms import (
            VIAL_REST_HEIGHT,
            _placement_values,
            bilateral_contact,
            contact_state,
            fingertip_positions_w,
            grasp_center_w,
            vial_grasp_point_w,
        )

        joint_position = _tensor(self.robot.data.joint_pos)
        joint_velocity = _tensor(self.robot.data.joint_vel)
        vial_position = _tensor(self.vial.data.root_pos_w)
        vial_quaternion = _tensor(self.vial.data.root_quat_w)
        vial_velocity = _tensor(self.vial.data.root_lin_vel_w)
        finite = torch.isfinite(joint_position).all(-1) & torch.isfinite(joint_velocity).all(-1)
        finite &= torch.isfinite(vial_position).all(-1) & torch.isfinite(vial_quaternion).all(-1)
        finite &= self.candidate_ik_valid
        limits = _tensor(self.robot.data.soft_joint_pos_limits)
        within_limits = ((joint_position >= limits[..., 0]) & (joint_position <= limits[..., 1])).all(-1)
        calm = joint_velocity.abs().amax(-1) < 2.0
        fixed, moving = fingertip_positions_w(self.env)
        contacts = contact_state(self.env)
        local, alignment, placement_speed, _, released, placed = _placement_values(self.env)
        vial_grasp_point = vial_grasp_point_w(self.env)
        from isaaclab.utils.math import quat_apply, quat_apply_inverse

        vial_axis = quat_apply(vial_quaternion, vial_position.new_tensor((0.0, 0.0, 1.0)).expand_as(vial_position))
        gripper_position = _tensor(self.robot.data.body_pos_w)[:, self.gripper_body_id].squeeze(1)
        gripper_quaternion = _tensor(self.robot.data.body_quat_w)[:, self.gripper_body_id].squeeze(1)
        grasp_offset = quat_apply_inverse(gripper_quaternion, grasp_center_w(self.env) - gripper_position)
        if phase == 3:
            # Servo-command fraction is not physical task progress: under a
            # finite-force load the vial can lag a nominally terminal joint
            # command. Label lift rows by the motion Newton actually reached
            # so a terminal-lift curriculum is genuinely adjacent to the
            # first reorientation states.
            lift_progress = measured_lift_progress(vial_position[:, 2], initial_vial_position[:, 2])
            self.candidate_difficulty.copy_((3.0 + lift_progress) / 7.0)
        self.last_diagnostics = {
            **self.candidate_diagnostics,
            "finite_rate": float(finite.float().mean()),
            "within_limits_rate": float(within_limits.float().mean()),
            "calm_rate": float(calm.float().mean()),
            "fixed_contact_rate": float(contacts[:, 0].mean()),
            "moving_contact_rate": float(contacts[:, 1].mean()),
            "bilateral_contact_rate": float(contacts[:, 2].mean()),
            "jaw_separation_mean_m": float(torch.linalg.vector_norm(fixed - moving, dim=-1).mean()),
            "grasp_offset_x_mean_m": float(grasp_offset[:, 0].mean()),
            "grasp_offset_y_mean_m": float(grasp_offset[:, 1].mean()),
            "grasp_offset_z_mean_m": float(grasp_offset[:, 2].mean()),
            "vial_height_mean_m": float((vial_position[:, 2] - self.env.scene.env_origins[:, 2]).mean()),
            "vial_speed_mean_mps": float(torch.linalg.vector_norm(vial_velocity, dim=-1).mean()),
            "gripper_position_mean_rad": float(joint_position[:, -1].mean()),
            "fixed_vial_distance_mean_m": float(torch.linalg.vector_norm(fixed - vial_grasp_point, dim=-1).mean()),
            "moving_vial_distance_mean_m": float(torch.linalg.vector_norm(moving - vial_grasp_point, dim=-1).mean()),
            "grasp_center_distance_mean_m": float(
                torch.linalg.vector_norm(vial_grasp_point - grasp_center_w(self.env), dim=-1).mean()
            ),
            "grasp_center_distance_rate": float(
                (
                    torch.linalg.vector_norm(vial_grasp_point - grasp_center_w(self.env), dim=-1)
                    < self.cfg.contact_distance
                )
                .float()
                .mean()
            ),
            "vial_angular_speed_mean_radps": float(
                torch.linalg.vector_norm(_tensor(self.vial.data.root_ang_vel_w), dim=-1).mean()
            ),
            "vial_angular_speed_rate": float(
                (torch.linalg.vector_norm(_tensor(self.vial.data.root_ang_vel_w), dim=-1) < 1.0).float().mean()
            ),
            "rack_local_x_mean_m": float(local[:, 0].mean()),
            "rack_local_y_mean_m": float(local[:, 1].mean()),
            "rack_local_z_mean_m": float(local[:, 2].mean()),
            "vertical_alignment_mean": float(alignment.mean()),
            "vial_axis_z_mean": float(vial_axis[:, 2].mean()),
            "vial_axis_z_min": float(vial_axis[:, 2].min()),
            "vial_axis_z_max": float(vial_axis[:, 2].max()),
            "vertical_alignment_rate": float((alignment > 0.88).float().mean()),
            "placed_rate": float(placed.float().mean()),
            "released_rate": float(released.float().mean()),
            "placement_speed_mean_mps": float(placement_speed.mean()),
            "placement_speed_rate": float((placement_speed < 0.15).float().mean()),
            "safe_trajectory_rate": float((~self.candidate_unsafe).float().mean()),
            "rack_contact_free_rate": float((~self.candidate_rack_contact).float().mean()),
            "clearance_valid_rate": float((~self.candidate_clearance_violation).float().mean()),
        }
        phase_progress = (self.candidate_difficulty * 7.0 - phase).clamp(0.0, 1.0)
        terminal = torch.isclose(phase_progress, torch.ones_like(phase_progress))
        self.last_diagnostics.update(
            {
                "phase_progress_mean": float(phase_progress.mean()),
                "terminal_candidate_count": float(terminal.sum()),
            }
        )
        best_alignment_row = alignment.argmax()
        self.last_diagnostics["best_alignment_progress"] = float(phase_progress[best_alignment_row])
        for bin_index in range(10):
            lower = bin_index / 10.0
            upper = (bin_index + 1) / 10.0
            in_bin = (phase_progress >= lower) & (phase_progress < upper if bin_index < 9 else phase_progress <= upper)
            if bool(in_bin.any()):
                self.last_diagnostics[f"alignment_max_progress_{lower:.1f}_{upper:.1f}"] = float(
                    alignment[in_bin].max()
                )
        if bool(terminal.any()):
            self.last_diagnostics.update(
                {
                    "terminal_alignment_mean": float(alignment[terminal].mean()),
                    "terminal_alignment_max": float(alignment[terminal].max()),
                    "terminal_bilateral_rate": float(contacts[terminal, 2].mean()),
                    "terminal_vial_speed_mean_mps": float(
                        torch.linalg.vector_norm(vial_velocity[terminal], dim=-1).mean()
                    ),
                }
            )
        if phase in (0, 1):
            local_position = vial_position - self.env.scene.env_origins
            drift = torch.linalg.vector_norm(vial_position - initial_vial_position, dim=-1)
            speed = torch.linalg.vector_norm(vial_velocity, dim=-1)
            table_stable = local_position[:, 2] > VIAL_REST_HEIGHT - 0.005
            table_stable &= local_position[:, 2] < VIAL_REST_HEIGHT + 0.006
            table_stable &= (speed < 0.08) & (drift < 0.006)
            if phase == 0:
                stable = table_stable & ~contacts[:, :2].any(-1)
            else:
                distance = torch.linalg.vector_norm(vial_grasp_point - grasp_center_w(self.env), dim=-1)
                angular_speed = torch.linalg.vector_norm(_tensor(self.vial.data.root_ang_vel_w), dim=-1)
                closed_grasp = bilateral_contact(self.env) & (distance < self.cfg.contact_distance)
                closed_grasp &= (local_position[:, 2] > 0.038) & (local_position[:, 2] < 0.080)
                closed_grasp &= (speed < 0.08) & (angular_speed < 0.8) & (alignment < 0.20) & (drift < 0.025)
                stable = (table_stable | closed_grasp) & (distance < 0.100)
                proof_segment = self.candidate_difficulty > PREGRASP_PROOF_DIFFICULTY
                proof_progress = (3.0 * (7.0 * self.candidate_difficulty - 1.0) - 2.0).clamp(0.0, 1.0)
                # During the supported pivot the root barely rises; the
                # cap-side grasp point is the part deliberately lifted.
                measured_lift = vial_grasp_point[:, 2] - initial_vial_position[:, 2]
                proof_valid = closed_grasp & (measured_lift > 0.006 * proof_progress)
                stable &= ~proof_segment | proof_valid
            self.last_diagnostics.update(
                {
                    "table_height_rate": float(
                        (
                            (local_position[:, 2] > VIAL_REST_HEIGHT - 0.005)
                            & (local_position[:, 2] < VIAL_REST_HEIGHT + 0.006)
                        )
                        .float()
                        .mean()
                    ),
                    "table_speed_rate": float((torch.linalg.vector_norm(vial_velocity, dim=-1) < 0.08).float().mean()),
                    "table_drift_rate": float((drift < 0.006).float().mean()),
                    "table_stable_rate": float(stable.float().mean()),
                    "closed_grasp_rate": float(bilateral_contact(self.env).float().mean()),
                }
            )
            return finite & within_limits & calm & stable
        if phase == 2:
            distance = torch.linalg.vector_norm(vial_grasp_point - grasp_center_w(self.env), dim=-1)
            drift = torch.linalg.vector_norm(vial_position - initial_vial_position, dim=-1)
            proof_lift = vial_grasp_point[:, 2] - initial_vial_position[:, 2]
            angular_speed = torch.linalg.vector_norm(_tensor(self.vial.data.root_ang_vel_w), dim=-1)
            stable_table = (vial_position[:, 2] - self.env.scene.env_origins[:, 2]) > 0.038
            stable_table &= torch.linalg.vector_norm(vial_velocity, dim=-1) < 0.08
            self.last_diagnostics["proof_lift_rate"] = float((proof_lift > 0.006).float().mean())
            self.last_diagnostics["proof_lift_mean_m"] = float(proof_lift.mean())
            self.last_diagnostics["horizontal_grasp_rate"] = float((alignment < 0.20).float().mean())
            self.last_diagnostics["proof_stable_table_rate"] = float(stable_table.float().mean())
            self.last_diagnostics["proof_angular_speed_rate"] = float((angular_speed < 0.8).float().mean())
            self.last_diagnostics["proof_centered_rate"] = float((distance < 0.030).float().mean())
            self.last_diagnostics["proof_drift_rate"] = float((drift < 0.025).float().mean())
            self.last_diagnostics["proof_drift_mean_m"] = float(drift.mean())
            proof_mask = self.candidate_ik_valid.clone()
            if bool(proof_mask.any()):
                self.last_diagnostics["proof_candidate_count"] = float(proof_mask.sum())
                proof_checks = {
                    "bilateral": bilateral_contact(self.env),
                    "stable_table": stable_table,
                    "lift": proof_lift > 0.006,
                    "horizontal": alignment < 0.20,
                    "calm_rotation": angular_speed < 0.8,
                    "centered": distance < 0.030,
                    "low_drift": drift < 0.025,
                }
                for name, check in proof_checks.items():
                    self.last_diagnostics[f"proof_valid_{name}_rate"] = float(check[proof_mask].float().mean())
            proof_valid = (
                finite
                & within_limits
                & calm
                & bilateral_contact(self.env)
                & stable_table
                & (proof_lift > 0.006)
                & (alignment < 0.20)
                & (angular_speed < 0.8)
                & (distance < 0.030)
                & (drift < 0.025)
            )
            self.last_diagnostics["proof_connected_count"] = float(proof_valid.sum())
            return proof_valid
        if phase == 7:
            _, alignment, speed, angular_speed, _, placed = _placement_values(self.env)
            return (
                finite
                & within_limits
                & calm
                & ~self.candidate_unsafe
                & placed
                # Match the task's physical success criterion.  The real
                # four-hole rack permits a few degrees of passive tilt after
                # release; demanding >0.94 rejected stable seated samples and
                # would teach the policy to keep pressing on the vial.
                & (alignment > 0.90)
                & (speed < 0.08)
                & (angular_speed < 1.0)
            )
        contact = bilateral_contact(self.env)
        distance = torch.linalg.vector_norm(vial_grasp_point - grasp_center_w(self.env), dim=-1)
        stable_grasp = contact
        if phase < 4:
            # The proof, lift, and reorientation phases must begin at the
            # intended cap/shoulder grasp. During reorientation the light vial
            # can slide axially while remaining physically enclosed by both
            # jaws. A distance to one authored point is no longer a valid
            # retention test after that slide; live bilateral vial contact and
            # the dynamic stability checks below are the physical criteria.
            stable_grasp &= distance < self.cfg.contact_distance
        stable_grasp &= placement_speed < 0.15
        stable_grasp &= torch.linalg.vector_norm(_tensor(self.vial.data.root_ang_vel_w), dim=-1) < 1.0
        if phase == 3:
            # Lift is its own physical milestone.  Do not require the vial to
            # follow an authored orientation schedule while it leaves the mat.
            stable_grasp &= vial_grasp_point[:, 2] > initial_vial_position[:, 2] + 0.006
        if phase == 4:
            progress = (self.candidate_difficulty * 7.0 - 4.0).clamp(0.0, 1.0)
            # Reorientation must make progress and finish upright, but the
            # dynamic vial is free to lag the commanded real-robot trajectory.
            stable_grasp &= alignment > (progress - 0.12).clamp_min(0.0)
        if phase == 5:
            # Keep the entire connected reorient/transport segment. Requiring
            # candidates to be upright and near the rack here would collapse
            # the reset distribution back to an endpoint cluster and leave the
            # policy no states on which to learn that transition.
            stable_grasp &= vial_position[:, 2] > 0.050
        if phase == 6:
            _, alignment, speed, _, _, _ = _placement_values(self.env)
            centered = (local[:, 0].abs() < 0.045) & (local[:, 1].abs() < 0.065)
            insertion_height = (local[:, 2] > 0.030) & (local[:, 2] < 0.130)
            stable_grasp &= centered & insertion_height & (alignment > 0.88) & (speed < 0.15)
        if phase >= 3:
            stable_grasp &= ~self.candidate_unsafe
        return finite & within_limits & calm & stable_grasp

    def _connected_candidate(self, phase: int, *, terminal_insert: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Reach a requested phase through one connected physical trajectory."""
        self.candidate_diagnostics = {}
        self.candidate_ik_valid.fill_(True)
        self.candidate_unsafe.fill_(False)
        self.candidate_rack_contact.fill_(False)
        self.candidate_clearance_violation.fill_(False)
        self._track_safety = False
        self._track_clearance = False
        seed_phase = 2
        for bank_name, bank_phase in (("transport", 5), ("reorient", 4), ("lift", 3), ("grasp", 2)):
            if phase > bank_phase and bank_name in self._seed_banks:
                target, initial_world_position = self._restore_seed_bank(bank_name)
                seed_phase = bank_phase
                break
        else:
            _, target, _, initial_world_position = self._approach_candidate(None)
            self._record_grasp_diagnostics("preclose")
            preclose_error = torch.linalg.vector_norm(
                _tensor(self.robot.data.joint_pos) - self._last_pregrasp_joint_position, dim=-1
            )
            preclose_mask = self.candidate_ik_valid & torch.isfinite(preclose_error)
            if bool(preclose_mask.any()):
                self.candidate_diagnostics["preclose_joint_error_mean_rad"] = float(
                    preclose_error[preclose_mask].mean()
                )
            close_target = target.clone()
            close_target[:, -1] = self.gripper_closed_position
            # Grasp reset states represent a completed, load-bearing close.
            # Pregrasp states already cover learning the closing transition.
            close_fraction = torch.ones(self.num_envs, device=self.device)
            target[:, -1] = torch.lerp(target[:, -1], close_target[:, -1], close_fraction)
            self._move_robot(target, self.cfg.grasp_close_steps)
            self._simulate(self.cfg.contact_settle_steps, target)
            self._record_grasp_diagnostics("closed")

            if phase == 2:
                # A bilateral table contact is not yet evidence of a usable
                # grasp. Raise the dynamic vial clear of its support before a
                # state may enter the dataset or seed the lift phase.
                proof_translation = target.new_tensor((-0.005, -0.002, 0.010)).expand(self.num_envs, -1)
                target = self._translate_grasped_vial(
                    target,
                    proof_translation,
                    torch.ones(self.num_envs, device=self.device),
                )
                self._simulate(self.cfg.contact_settle_steps, target)
                self._record_grasp_diagnostics("proof")

        segment_fraction = torch.ones(self.num_envs, device=self.device)
        for next_phase in range(seed_phase + 1, min(phase, 6) + 1):
            if next_phase == phase and not (terminal_insert and phase == 6):
                segment_fraction = torch.empty(self.num_envs, device=self.device).uniform_(
                    0.02, 0.98, generator=self.random
                )
                # Randomize terminal rows instead of tying them to fixed
                # environment indices. Asset randomization and restored grasp
                # seeds can make some indices consistently more viable than
                # others, and every viable endpoint must have a chance to seed
                # the next connected phase.
                terminal_rows = torch.rand(self.num_envs, device=self.device, generator=self.random) < 0.5
                terminal_rows[-1] = True
                segment_fraction[terminal_rows] = 1.0
            else:
                segment_fraction = torch.ones(self.num_envs, device=self.device)
            # A viable reset must have a collision-free history, not merely a
            # calm endpoint. Track unintended rack contact from the first
            # loaded lift waypoint through insertion.
            if next_phase >= 3:
                self._track_safety = True
                self._track_clearance = False
            if next_phase == 5:
                self._track_safety = True
                self._track_clearance = False
                target = self._transport_vial(
                    target,
                    self._desired_vial_pose(5),
                    segment_fraction,
                )
            elif next_phase == 6:
                self._track_safety = True
                self._track_clearance = True
                insertion_target = self._solve_vial_pose(self._desired_vial_pose(6), target)
                target = torch.lerp(target, insertion_target, segment_fraction.unsqueeze(-1))
                self._move_robot(target, 2 * self.cfg.transition_steps)
                self._simulate(self.cfg.contact_settle_steps // 3, target)
            else:
                target = self._follow_workshop_segment(target, next_phase, segment_fraction)
                if next_phase == 4:
                    # The sparse real waypoint reaches the upright basin, but
                    # finite drive deflection leaves a repeatable residual
                    # tilt in Newton. Close that error with measured Cartesian
                    # feedback so later reset branches can physically enter
                    # the slot. The vial remains fully dynamic throughout.
                    terminal_rows = torch.isclose(segment_fraction, torch.ones_like(segment_fraction))
                    target = self._correct_upright_endpoint(target, terminal_rows)
            if phase > next_phase:
                self.candidate_difficulty = (next_phase + segment_fraction) / 7.0
                valid_seed = self._valid(next_phase, initial_world_position)
                bank_name = {3: "lift", 4: "reorient", 5: "transport"}.get(next_phase)
                if bank_name is not None:
                    self._append_seed_bank(bank_name, valid_seed, target)

        if phase == 7:
            segment_fraction = torch.empty(self.num_envs, device=self.device).uniform_(0.0, 1.0, generator=self.random)
            target = self._follow_workshop_segment(target, 7, segment_fraction)
        self.candidate_difficulty = (phase + segment_fraction).clamp_max(7.0) / 7.0
        return self._valid(phase, initial_world_position), target

    def _release_candidate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Branch a physical jaw-opening rollout from validated insertion states."""
        count = self.release_seed_joint_position.shape[0]
        rows = torch.randint(count, (self.num_envs,), device=self.device, generator=self.random)
        joint_position = self.release_seed_joint_position[rows]
        target = self.release_seed_joint_target[rows].clone()
        local_pose = self.release_seed_vial_pose[rows]
        self._write_robot(joint_position)
        self._write_vial(local_pose)
        self.robot.set_joint_position_target_index(target=target)
        self.action_term._joint_target.copy_(target)
        self._commit_reset_state()
        self._track_safety = True
        self._track_clearance = True
        self._simulate(self.cfg.contact_settle_steps // 3, target)

        initial_world_position = local_pose[:, :3] + self.env.scene.env_origins
        segment_fraction = torch.empty(self.num_envs, device=self.device).uniform_(0.0, 1.0, generator=self.random)
        release_target = target.clone()
        release_target[:, -1] = RELEASE_GRIPPER_POSITION
        target = torch.lerp(target, release_target, segment_fraction.unsqueeze(-1))
        self._move_robot(target, self.cfg.grasp_close_steps)
        # Give the released vial time to traverse the real slot under gravity.
        # The old 0.75 s check accepted rim engagement before seating.
        self._simulate(2 * self.cfg.contact_settle_steps, target)
        self.candidate_difficulty.fill_(1.0)
        return self._valid(7, initial_world_position), target

    def _candidate_batch(self, phase: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._reset_rack()
        self.candidate_diagnostics = {}
        self.candidate_ik_valid.fill_(True)
        self.candidate_unsafe.fill_(False)
        self.candidate_rack_contact.fill_(False)
        self.candidate_clearance_violation.fill_(False)
        self._track_safety = False
        self._track_clearance = False
        if phase == 7 and not hasattr(self, "release_seed_joint_position"):
            valid, target = self._connected_candidate(6, terminal_insert=True)
            # Release branches must begin from the same physically verified
            # held insertion used online. A centered but tilted vial can lodge
            # on the tight rim and never descend after the jaws open.
            from ..mdp.terms import held_insertion_ready

            valid &= held_insertion_ready(self.env)
            if not valid.any():
                return valid, target
            self.release_seed_joint_position = _tensor(self.robot.data.joint_pos)[valid].detach().clone()
            self.release_seed_joint_target = target[valid].detach().clone()
            self.release_seed_vial_pose = torch.cat(
                (
                    _tensor(self.vial.data.root_pos_w)[valid] - self.env.scene.env_origins[valid],
                    _tensor(self.vial.data.root_quat_w)[valid],
                ),
                dim=-1,
            ).detach()
        if phase == 7:
            self.candidate_ik_valid.fill_(True)
            self.candidate_unsafe.fill_(False)
            self.candidate_rack_contact.fill_(False)
            self.candidate_clearance_violation.fill_(False)
            self._track_safety = False
            self._track_clearance = False
            return self._release_candidate()
        if phase >= 2:
            valid, target = self._connected_candidate(phase)
            if phase == 2:
                # Exercise the exact serialization path used by the reset
                # dataset without broadcasting one lucky proposal over the
                # batch. Every accepted row remains a distinct dynamically
                # reached grasp and must survive its own zero-velocity restore.
                connected_valid = valid.clone()
                connected_diagnostics = self.last_diagnostics.copy()
                joint_position = _tensor(self.robot.data.joint_pos).detach().clone()
                local_pose = torch.cat(
                    (
                        _tensor(self.vial.data.root_pos_w) - self.env.scene.env_origins,
                        _tensor(self.vial.data.root_quat_w),
                    ),
                    dim=-1,
                ).detach()
                self._write_robot(joint_position)
                self._write_vial(local_pose)
                self.robot.set_joint_position_target_index(target=target)
                self.action_term._joint_target.copy_(target)
                self._commit_reset_state()
                self.candidate_ik_valid.fill_(True)
                self._simulate(self.cfg.contact_settle_steps, target)
                # Reconstruct the table-rest reference used by the proof-lift
                # test. The restored pose itself must remain at least 6 mm
                # above it while carrying the vial at zero initial velocity.
                initial_world_position = local_pose[:, :3] + self.env.scene.env_origins
                initial_world_position[:, 2] -= 0.012
                restore_valid = self._valid(phase, initial_world_position)
                for key, value in connected_diagnostics.items():
                    self.last_diagnostics[f"connected_{key}"] = value
                self.last_diagnostics["connected_valid_count"] = float(connected_valid.sum())
                self.last_diagnostics["restore_valid_count"] = float(restore_valid.sum())
                if bool(connected_valid.any()):
                    from ..mdp.terms import bilateral_contact, grasp_center_w, vial_grasp_point_w

                    restore_contact = bilateral_contact(self.env)
                    restore_distance = torch.linalg.vector_norm(
                        grasp_center_w(self.env) - vial_grasp_point_w(self.env), dim=-1
                    )
                    self.last_diagnostics["restore_connected_bilateral_rate"] = float(
                        restore_contact[connected_valid].float().mean()
                    )
                    self.last_diagnostics["restore_connected_centered_rate"] = float(
                        (restore_distance[connected_valid] < 0.030).float().mean()
                    )
                # A serialized row must retain both proofs: the connected
                # dynamic rollout lifted the vial, and the exact restored
                # snapshot remains a stable bilateral grasp. Never replace
                # the real proof with the synthetic restore reference alone.
                valid = connected_valid & restore_valid
            seed_name = {2: "grasp", 3: "lift", 4: "reorient", 5: "transport"}.get(phase)
            if seed_name is not None:
                terminal = valid
                if phase in (3, 4, 5):
                    terminal_difficulty = (phase + 1.0) / 7.0
                    terminal = valid & torch.isclose(
                        self.candidate_difficulty,
                        torch.full_like(self.candidate_difficulty, terminal_difficulty),
                    )
                if phase == 4:
                    # Intermediate reorientation rows may legitimately lag
                    # the command. Only near-upright endpoints may seed
                    # transport; later segments preserve this orientation.
                    from ..mdp.geometry import vertical_alignment
                    from ..mdp.terms import HELD_INSERTION_ALIGNMENT

                    terminal &= (
                        vertical_alignment(_tensor(self.vial.data.root_quat_w)) > HELD_INSERTION_ALIGNMENT
                    )
                self._append_seed_bank(seed_name, terminal, target)
            return valid, target
        valid, target, _, _ = self._approach_candidate(phase)
        return valid, target

    def generate(self) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
        """Generate balanced phases with bounded rejection sampling."""
        rows = {
            name: []
            for name in ("joint_position", "joint_target", "vial_pose", "phase", "difficulty", "grasped", "lifted")
        }
        for phase, phase_name in enumerate(PHASE_NAMES):
            # Each transition bank must be fed by independently generated
            # physical roots, beginning with the grasp itself. Otherwise a
            # large batched rollout merely clones one lucky grasp and cannot
            # explore downstream mechanical outcomes.
            required_seed = {2: "grasp", 3: "lift", 4: "reorient", 5: "transport"}.get(phase)

            def branch_ready(name: str | None = required_seed) -> bool:
                return name is None or (
                    name in self._seed_banks
                    and self._seed_banks[name]["joint_position"].shape[0] >= self.cfg.branch_seed_count
                )

            accepted = 0
            attempts = 0
            invalid = 0
            surplus_valid = 0
            while (
                accepted < self.cfg.poses_per_phase or not branch_ready()
            ) and attempts < self.cfg.max_attempts_per_phase:
                valid, joint_target = self._candidate_batch(phase)
                attempts += self.num_envs
                ids = valid.nonzero(as_tuple=False).squeeze(-1)
                invalid += self.num_envs - ids.numel()
                take = ids[: self.cfg.poses_per_phase - accepted]
                surplus_valid += ids.numel() - take.numel()
                if attempts % (4 * self.num_envs) == 0 and (
                    accepted < self.cfg.poses_per_phase or not branch_ready()
                ):
                    summary = {
                        key: self.last_diagnostics[key]
                        for key in (
                            "ik_valid_rate",
                            "pregrasp_ik_valid_rate",
                            "pregrasp_position_rate",
                            "pregrasp_closing_rate",
                            "pregrasp_pad_rate",
                            "pregrasp_margin_rate",
                            "pregrasp_live_tcp_error_mean_m",
                            "pregrasp_live_vial_drift_mean_m",
                            "settle_command_saturation_rate",
                            "settle_joint_error_mean_rad",
                            "teleport_fixed_distance_mean_m",
                            "teleport_fixed_closing_offset_mean_m",
                            "teleport_moving_closing_offset_mean_m",
                            "preclose_center_error_mean_m",
                            "preclose_joint_error_mean_rad",
                            "preclose_fixed_closing_offset_mean_m",
                            "preclose_moving_closing_offset_mean_m",
                            "closed_bilateral_rate",
                            "closed_center_error_mean_m",
                            "closed_fixed_closing_offset_mean_m",
                            "closed_moving_closing_offset_mean_m",
                            "proof_bilateral_rate",
                            "proof_center_error_mean_m",
                            "proof_valid_bilateral_rate",
                            "proof_valid_stable_table_rate",
                            "proof_valid_lift_rate",
                            "proof_valid_horizontal_rate",
                            "proof_valid_calm_rotation_rate",
                            "proof_valid_centered_rate",
                            "proof_valid_low_drift_rate",
                            "proof_connected_count",
                            "connected_finite_rate",
                            "connected_within_limits_rate",
                            "connected_calm_rate",
                            "connected_proof_valid_bilateral_rate",
                            "connected_proof_valid_stable_table_rate",
                            "connected_proof_valid_lift_rate",
                            "connected_proof_valid_horizontal_rate",
                            "connected_proof_valid_calm_rotation_rate",
                            "connected_proof_valid_centered_rate",
                            "connected_proof_valid_low_drift_rate",
                            "connected_proof_lift_mean_m",
                            "connected_proof_drift_mean_m",
                            "connected_vial_speed_mean_mps",
                            "reorient_endpoint_valid_rate",
                            "connected_valid_count",
                            "restore_valid_count",
                            "restore_connected_bilateral_rate",
                            "restore_connected_centered_rate",
                            "bilateral_contact_rate",
                            "grasp_center_distance_mean_m",
                            "grasp_center_distance_rate",
                            "vial_height_mean_m",
                            "vial_speed_mean_mps",
                            "placement_speed_rate",
                            "vial_angular_speed_mean_radps",
                            "vial_angular_speed_rate",
                            "vertical_alignment_mean",
                            "vial_axis_z_max",
                            "safe_trajectory_rate",
                        )
                        if key in self.last_diagnostics
                    }
                    seed_count = (
                        self._seed_banks[required_seed]["joint_position"].shape[0]
                        if required_seed in self._seed_banks
                        else 0
                    )
                    print(
                        f"[INFO] {phase_name}: progress accepted={accepted}, attempted={attempts}, "
                        f"seed_count={seed_count}, diagnostics={summary}",
                        flush=True,
                    )
                if take.numel() == 0:
                    continue
                joint_position = _tensor(self.robot.data.joint_pos)[take].detach().clone()
                vial_pose = torch.cat(
                    (
                        _tensor(self.vial.data.root_pos_w)[take] - self.env.scene.env_origins[take],
                        _tensor(self.vial.data.root_quat_w)[take],
                    ),
                    dim=-1,
                ).detach()
                count = take.numel()
                rows["joint_position"].append(joint_position)
                rows["joint_target"].append(joint_target[take].detach().clone())
                rows["vial_pose"].append(vial_pose)
                rows["phase"].append(torch.full((count,), phase, device=self.device, dtype=torch.long))
                rows["difficulty"].append(self.candidate_difficulty[take].detach().clone())
                from ..mdp.terms import bilateral_contact

                live_grasp = bilateral_contact(self.env)[take]
                represented_grasp = live_grasp | (phase >= 3)
                rack_z = _tensor(self.rack.data.default_root_pose)[take, 2]
                represented_lift = _represented_lift(phase, vial_pose, rack_z, represented_grasp)
                rows["grasped"].append(represented_grasp.detach().clone())
                rows["lifted"].append(represented_lift.detach().clone())
                accepted += count
            self.rejections[phase_name] = invalid
            seed_missing = not branch_ready()
            if accepted != self.cfg.poses_per_phase or seed_missing:
                seed_detail = f"; no viable {required_seed!r} branch endpoint" if seed_missing else ""
                raise RuntimeError(
                    f"Generated only {accepted}/{self.cfg.poses_per_phase} valid {phase_name!r} poses "
                    f"after {attempts} attempts{seed_detail}. Last batch diagnostics: {self.last_diagnostics}."
                )
            seed_counts = {name: bank["joint_position"].shape[0] for name, bank in self._seed_banks.items()}
            print(
                f"[INFO] {phase_name}: accepted={accepted}, invalid={invalid}, "
                f"surplus_valid={surplus_valid}, attempted={attempts}, seed_banks={seed_counts}",
                flush=True,
            )
        states = {name: torch.cat(parts, dim=0) for name, parts in rows.items()}
        return states, self.rejections


def generate_main(argv: list[str] | None = None) -> int:
    """CLI entry point for reset generation."""
    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser(description="Generate physics-validated SO-101 vial reset poses.")
    parser.add_argument("--output", type=Path, default=RESET_DATASET)
    parser.add_argument("--poses_per_phase", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_attempts_per_phase", type=int, default=32_768)
    parser.add_argument("--branch_seed_count", type=int, default=32)
    add_launcher_args(parser)
    args = parser.parse_args(argv)

    from isaaclab.envs import ManagerBasedRLEnv

    from ..env_cfg import SO101VialGeneratorEnvCfg

    cfg = GeneratorCfg(
        poses_per_phase=args.poses_per_phase,
        batch_size=args.batch_size,
        seed=args.seed,
        max_attempts_per_phase=args.max_attempts_per_phase,
        branch_seed_count=args.branch_seed_count,
    )
    env_cfg = SO101VialGeneratorEnvCfg()
    env_cfg.scene.num_envs = cfg.batch_size
    env_cfg.sim.device = args.device
    env_cfg.seed = cfg.seed
    with launch_simulation(env_cfg, args):
        env = ManagerBasedRLEnv(env_cfg)
        try:
            states, rejections = _Generator(env, cfg).generate()
        finally:
            env.close()
    artifact = save_reset_dataset(
        args.output,
        states,
        generator=asdict(cfg),
        validation={
            "physics": "newton_mjwarp",
            "gravity": True,
            "object_state_writes_after_reset": False,
            "rejections": rejections,
        },
    )
    print(f"[INFO] Wrote {artifact['row_count']} reset poses to {Path(args.output).resolve()}.")
    print(f"[INFO] Content SHA-256: {artifact['content_sha256']}")
    return 0


def view_main(argv: list[str] | None = None) -> int:
    """Cycle a reset artifact in the Newton visualizer without a policy."""
    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser(description="Cycle SO-101 reset poses in the Newton visualizer.")
    parser.add_argument("--dataset", type=Path, default=RESET_DATASET)
    parser.add_argument("--steps_per_pose", type=int, default=45)
    parser.add_argument("--cycles", type=int, default=1)
    add_launcher_args(parser)
    args = parser.parse_args(argv)

    from isaaclab.envs import ManagerBasedRLEnv

    from ..env_cfg import SO101VialGeneratorEnvCfg

    env_cfg = SO101VialGeneratorEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args.device
    with launch_simulation(env_cfg, args):
        env = ManagerBasedRLEnv(env_cfg)
        try:
            artifact = load_reset_dataset(args.dataset, device=env.device)
            states = artifact["states"]
            order = torch.argsort(states["phase"].float() + states["difficulty"] * 0.01, stable=True)
            robot = env.scene["robot"]
            vial = env.scene["vial"]
            zeros = torch.zeros((1, 6), device=env.device)
            action_term = env.action_manager.get_term("joint_delta")
            for _ in range(args.cycles):
                for row in order.tolist():
                    joint_position = states["joint_position"][row : row + 1]
                    joint_target = states["joint_target"][row : row + 1]
                    vial_pose = states["vial_pose"][row : row + 1].clone()
                    vial_pose[:, :3] += env.scene.env_origins[:1]
                    robot.write_joint_position_to_sim_index(position=joint_position)
                    robot.write_joint_velocity_to_sim_index(velocity=zeros)
                    vial.write_root_pose_to_sim_index(root_pose=vial_pose)
                    vial.write_root_velocity_to_sim_index(root_velocity=zeros)
                    rack = env.scene["rack"]
                    rack_pose = _tensor(rack.data.default_root_pose).clone()
                    rack_pose[:, :3] += env.scene.env_origins
                    rack.write_root_pose_to_sim_index(root_pose=rack_pose)
                    rack.write_root_velocity_to_sim_index(root_velocity=zeros)
                    action_term._joint_target.copy_(joint_target)
                    print(
                        f"[RESET] row={row} phase={PHASE_NAMES[int(states['phase'][row])]} "
                        f"difficulty={float(states['difficulty'][row]):.3f}",
                        flush=True,
                    )
                    for _ in range(args.steps_per_pose):
                        if not env.sim.is_headless_or_exist_active_visualizer():
                            return 0
                        robot.set_joint_position_target_index(target=joint_target)
                        env.scene.write_data_to_sim()
                        env.sim.step()
                        env.scene.update(env.physics_dt)
        finally:
            env.close()
    return 0
