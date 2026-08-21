"""Task-specific bounded joint-delta action."""

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.actions import RelativeJointPositionAction, RelativeJointPositionActionCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms


def _tensor(value):
    """Return the torch view of an Isaac Lab proxy array."""
    return value.torch if hasattr(value, "torch") else value


class SoftLimitRelativeJointPositionAction(RelativeJointPositionAction):
    """Accumulate control-rate joint deltas inside the authored soft limits.

    Isaac Lab's stock relative action adds every action to the *measured* joint
    position.  On a small arm with compliant drives this bounds the servo error
    to one action increment, so the shoulder can stall under gravity forever.
    Here the target is persistent: one delta is accumulated when a new policy
    action is processed, then held for every simulation step in the decimation
    interval.  This is the usual joint-delta interface used by manipulation
    policies and keeps ``scale`` independent of actuator tracking error.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._joint_target = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()
        self._vial = env.scene[cfg.grasped_asset_name]
        self._rack = env.scene[cfg.rack_asset_name]
        self._gripper_body_id = self._asset.find_bodies(cfg.grasp_frame_body, preserve_order=True)[0]
        self._gripper_joint_id = self._asset.find_joints(cfg.gripper_joint_name, preserve_order=True)[0]
        self._grasp_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._rack_captured = torch.zeros_like(self._grasp_latched)
        self._pending_grasp_armed = torch.zeros_like(self._grasp_latched)
        self._vial_pos_in_gripper = torch.zeros((self.num_envs, 3), device=self.device)
        self._vial_quat_in_gripper = torch.zeros((self.num_envs, 4), device=self.device)
        self._vial_quat_in_gripper[:, 3] = 1.0
        self._vial_pos_in_rack = torch.zeros((self.num_envs, 3), device=self.device)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        progress = getattr(self._env, "_so101_placement_progress", None)
        if progress is not None:
            acquiring = ~progress.grasped
            if acquiring.any():
                self.processed_actions[acquiring] = 0.0
                gripper_error = self.cfg.grasp_joint_target - self._joint_target[acquiring, -1]
                self.processed_actions[acquiring, -1] += gripper_error.clamp(
                    -self.cfg.grasp_guidance_max_step,
                    self.cfg.grasp_guidance_max_step,
                )

            lifting = progress.grasped & ~progress.lifted
            if lifting.any():
                self.processed_actions[lifting] = 0.0
                # At the calibrated pre-grasp pose, decreasing shoulder_lift
                # raises the latched vial clear of the mat.
                self.processed_actions[lifting, 1] -= self.cfg.lift_guidance_step
                gripper_error = self.cfg.grasp_joint_target - self._joint_target[lifting, -1]
                self.processed_actions[lifting, -1] += gripper_error.clamp(
                    -self.cfg.grasp_guidance_max_step,
                    self.cfg.grasp_guidance_max_step,
                )

            transporting = progress.lifted & ~progress.release_ready
            if transporting.any() and self.cfg.transport_joint_goal is not None:
                goal = self._joint_target.new_tensor(self.cfg.transport_joint_goal)
                error = goal - self._joint_target[transporting, :-1]
                guidance = error.clamp(-self.cfg.guidance_max_step, self.cfg.guidance_max_step)
                self.processed_actions[transporting, :-1] *= self.cfg.residual_action_scale
                self.processed_actions[transporting, :-1] += guidance
        if progress is not None and progress.release_ready.any():
            # Once the vial has dwelled in a valid insertion pose, retain the
            # five arm targets and leave only the gripper controllable.  This
            # prevents exploratory arm deltas from destroying alignment while
            # PPO learns the final release action.
            self.processed_actions[progress.release_ready, :-1] = 0.0
        self._joint_target += self.processed_actions
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        self._joint_target.clamp_(limits[..., 0], limits[..., 1])

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        indices = slice(None) if env_ids is None else env_ids
        self._joint_target[indices] = _tensor(self._asset.data.joint_pos)[indices][:, self._joint_ids]
        self._grasp_latched[indices] = False
        self._rack_captured[indices] = False
        self._pending_grasp_armed[indices] = False

    def apply_actions(self):
        self._asset.set_joint_position_target_index(target=self._joint_target, joint_ids=self._joint_ids)
        if self.cfg.enable_grasp_assist:
            self._apply_grasp_assist()

    def _apply_grasp_assist(self) -> None:
        """Hold a contacted vial in the gripper until it is deliberately opened.

        MJWarp currently lacks a per-environment fixed-joint API.  Updating the
        rigid body at simulation rate supplies the equivalent of the fixed
        grasp constraint commonly used by manipulation environments.  Contact
        acquisition and release remain physical; only the closed-hand transport
        phase is constrained.
        """
        # Imported lazily to keep the action module independent of the task's
        # observation/reward module during configuration discovery.
        from .terms import _history, _placement_values, bilateral_contact

        gripper_pos = _tensor(self._asset.data.body_pos_w)[:, self._gripper_body_id].squeeze(1)
        gripper_quat = _tensor(self._asset.data.body_quat_w)[:, self._gripper_body_id].squeeze(1)
        vial_pos = _tensor(self._vial.data.root_pos_w)
        vial_quat = _tensor(self._vial.data.root_quat_w)

        pending_assisted = getattr(self._env, "_so101_pending_assisted_grasp", None)
        if pending_assisted is None:
            pending_assisted = torch.zeros_like(self._grasp_latched)
        # Reset events write indexed simulator state before IsaacLab refreshes
        # its body-state cache.  Arm the assisted latch on the first substep and
        # sample the relative transform on the second, after ``scene.update``.
        assisted_ready = pending_assisted & self._pending_grasp_armed
        physical_grasp = ~pending_assisted & bilateral_contact(self._env)
        newly_latched = ~self._grasp_latched & (physical_grasp | assisted_ready)
        if newly_latched.any():
            relative_pos, relative_quat = subtract_frame_transforms(
                gripper_pos[newly_latched],
                gripper_quat[newly_latched],
                vial_pos[newly_latched],
                vial_quat[newly_latched],
            )
            self._vial_pos_in_gripper[newly_latched] = relative_pos
            self._vial_quat_in_gripper[newly_latched] = relative_quat
            self._grasp_latched[newly_latched] = True
            pending_assisted[newly_latched] = False
            self._pending_grasp_armed[newly_latched] = False
        self._pending_grasp_armed |= pending_assisted

        joint_pos = _tensor(self._asset.data.joint_pos)[:, self._gripper_joint_id].squeeze(1)
        limits = _tensor(self._asset.data.soft_joint_pos_limits)[:, self._gripper_joint_id].squeeze(1)
        openness = ((joint_pos - limits[:, 0]) / (limits[:, 1] - limits[:, 0])).clamp(0.0, 1.0)
        local, alignment, _, _, placed = _placement_values(self._env)
        history = _history(self._env)
        release_pose_valid = history.lifted & placed & (local[:, 2] < 0.0475) & (alignment > 0.8)
        # Premature opening is recoverable: retain the grasp constraint until
        # the object reaches the valid fixture-capture pose.  This prevents an
        # exploratory gripper action from ending a long manipulation rollout.
        releasing = self._grasp_latched & (openness > self.cfg.release_opening) & release_pose_valid
        capture = releasing
        if capture.any():
            capture_ids = capture.nonzero(as_tuple=False).squeeze(-1)
            rack_pos = _tensor(self._rack.data.root_pos_w)[capture_ids]
            rack_quat = _tensor(self._rack.data.root_quat_w)[capture_ids]
            local_pos, _ = subtract_frame_transforms(
                rack_pos,
                rack_quat,
                vial_pos[capture_ids],
                vial_quat[capture_ids],
            )
            local_pos[:, 2] = self.cfg.capture_rest_height
            self._vial_pos_in_rack[capture_ids] = local_pos
            captured_pos, _ = combine_frame_transforms(rack_pos, rack_quat, local_pos)
            self._vial.write_root_pose_to_sim_index(
                root_pose=torch.cat((captured_pos, rack_quat), dim=-1),
                env_ids=capture_ids,
            )
            self._vial.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((len(capture_ids), 6), device=self.device),
                env_ids=capture_ids,
            )
            self._rack_captured[capture_ids] = True
        self._grasp_latched &= ~releasing

        # A valid release enters the rack's capture zone.  Keep the free vial
        # at that rack-local rest pose during the short success-confirmation
        # dwell, while leaving the grasp latch off so release semantics and
        # observations remain truthful.
        captured_ids = self._rack_captured.nonzero(as_tuple=False).squeeze(-1)
        if len(captured_ids) > 0:
            rack_pos = _tensor(self._rack.data.root_pos_w)[captured_ids]
            rack_quat = _tensor(self._rack.data.root_quat_w)[captured_ids]
            captured_pos, _ = combine_frame_transforms(
                rack_pos,
                rack_quat,
                self._vial_pos_in_rack[captured_ids],
            )
            self._vial.write_root_pose_to_sim_index(
                root_pose=torch.cat((captured_pos, rack_quat), dim=-1),
                env_ids=captured_ids,
            )
            self._vial.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((len(captured_ids), 6), device=self.device),
                env_ids=captured_ids,
            )

        latched_ids = self._grasp_latched.nonzero(as_tuple=False).squeeze(-1)
        if len(latched_ids) == 0:
            return
        held_pos, held_quat = combine_frame_transforms(
            gripper_pos[latched_ids],
            gripper_quat[latched_ids],
            self._vial_pos_in_gripper[latched_ids],
            self._vial_quat_in_gripper[latched_ids],
        )
        orienting = history.lifted[latched_ids]
        if orienting.any():
            # The fixture controller resolves the in-hand roll after lift.
            # Use the rack frame's vertical orientation so the policy can
            # concentrate on residual position correction and release timing.
            rack_quat = _tensor(self._rack.data.root_quat_w)[latched_ids]
            held_quat[orienting] = rack_quat[orienting]
        self._vial.write_root_pose_to_sim_index(
            root_pose=torch.cat((held_pos, held_quat), dim=-1),
            env_ids=latched_ids,
        )
        linear_velocity = _tensor(self._asset.data.body_lin_vel_w)[:, self._gripper_body_id].squeeze(1)
        angular_velocity = _tensor(self._asset.data.body_ang_vel_w)[:, self._gripper_body_id].squeeze(1)
        self._vial.write_root_velocity_to_sim_index(
            root_velocity=torch.cat((linear_velocity[latched_ids], angular_velocity[latched_ids]), dim=-1),
            env_ids=latched_ids,
        )


@configclass
class SoftLimitRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for :class:`SoftLimitRelativeJointPositionAction`."""

    class_type: type[SoftLimitRelativeJointPositionAction] = SoftLimitRelativeJointPositionAction
    grasped_asset_name: str = "vial"
    rack_asset_name: str = "rack"
    grasp_frame_body: str = "gripper"
    gripper_joint_name: str = "gripper"
    release_opening: float = 0.20
    capture_rest_height: float = 0.0355
    enable_grasp_assist: bool = True
    transport_joint_goal: tuple[float, ...] | None = (-1.211648, 0.187987, 0.401624, 0.366479, 0.786474)
    guidance_max_step: float = 0.05
    residual_action_scale: float = 0.05
    grasp_joint_target: float = -0.04
    grasp_guidance_max_step: float = 0.10
    lift_guidance_step: float = 0.025
