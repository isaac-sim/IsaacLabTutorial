"""Joint-space generator action and calibrated policy gripper action."""

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.actions import RelativeJointPositionAction, RelativeJointPositionActionCfg
from isaaclab.utils.configclass import configclass

from ..config.so101.control import GRIPPER_ACTION_THRESHOLD


def _tensor(value):
    """Return the torch view of an Isaac Lab proxy array."""
    return value.torch if hasattr(value, "torch") else value


class SoftLimitRelativeJointPositionAction(RelativeJointPositionAction):
    """Apply measured-relative arm commands and a binary gripper command.

    The arm follows Isaac Lab's relative joint-position action.  The final
    gripper scalar follows its standard binary convention: negative closes and
    non-negative opens to calibrated ordinary position targets.  It only
    writes robot targets, so every object motion remains a contact outcome.
    """

    def __init__(self, cfg: "SoftLimitRelativeJointPositionActionCfg", env):
        super().__init__(cfg, env)
        self._joint_target = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()
        self._gripper_index = self._joint_names.index("gripper")
        self._gripper_closed = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Sanitize and scale one policy command."""
        # A failed physics row is terminated and reset in the same control
        # cycle. Do not let one non-finite network sample reach the controller.
        super().process_actions(torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Restore a dataset preload, or start from the measured reset position."""
        super().reset(env_ids)
        indices = slice(None) if env_ids is None else env_ids
        measured = _tensor(self._asset.data.joint_pos)[indices][:, self._joint_ids]
        reset_target = getattr(self._env, "_so101_reset_joint_target", None)
        use_reset_target = getattr(self._env, "_so101_use_reset_joint_target", None)
        if reset_target is None or use_reset_target is None:
            self.seed_joint_target(indices, measured)
            return
        selected_target = reset_target[indices]
        selected_use = use_reset_target[indices].unsqueeze(-1)
        self.seed_joint_target(indices, torch.where(selected_use, selected_target, measured))

    def seed_joint_target(self, env_ids, joint_target: torch.Tensor) -> None:
        """Synchronize a generated reset target and its gripper latch."""
        self._joint_target[env_ids] = joint_target
        gripper_target = joint_target[:, self._gripper_index]
        close_distance = (gripper_target - self.cfg.gripper_close_position).abs()
        open_distance = (gripper_target - self.cfg.gripper_open_position).abs()
        self._gripper_closed[env_ids] = close_distance <= open_distance

    def apply_actions(self) -> None:
        """Apply relative arm targets and an absolute open/close jaw target."""
        self._joint_target.copy_(_tensor(self._asset.data.joint_pos)[:, self._joint_ids])
        self._joint_target.add_(self.processed_actions)
        gripper_action = self.raw_actions[:, self._gripper_index]
        self._gripper_closed.copy_(
            torch.where(
                gripper_action < -GRIPPER_ACTION_THRESHOLD,
                True,
                torch.where(gripper_action > GRIPPER_ACTION_THRESHOLD, False, self._gripper_closed),
            )
        )
        self._joint_target[:, self._gripper_index] = torch.where(
            self._gripper_closed,
            self._joint_target.new_tensor(self.cfg.gripper_close_position),
            self._joint_target.new_tensor(self.cfg.gripper_open_position),
        )
        limits = _tensor(self._asset.data.soft_joint_pos_limits)[:, self._joint_ids]
        self._joint_target.clamp_(limits[..., 0], limits[..., 1])
        self._asset.set_joint_position_target_index(target=self._joint_target, joint_ids=self._joint_ids)


@configclass
class SoftLimitRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for :class:`SoftLimitRelativeJointPositionAction`."""

    class_type: type[SoftLimitRelativeJointPositionAction] = SoftLimitRelativeJointPositionAction
    gripper_open_position: float = 0.0
    gripper_close_position: float = 0.0


class SoftLimitRelativeGripperAction(RelativeJointPositionAction):
    """Apply a bounded incremental jaw-position command.

    This preserves the real controller's ordinary position interface while
    avoiding hidden binary latch state in the policy action. Negative closes,
    positive opens, and zero holds the measured jaw position.
    """

    def process_actions(self, actions: torch.Tensor) -> None:
        """Sanitize the normalized policy command before scaling it."""
        super().process_actions(torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0))

    def apply_actions(self) -> None:
        """Apply the relative target without crossing authored soft limits."""
        target = _tensor(self._asset.data.joint_pos)[:, self._joint_ids] + self.processed_actions
        limits = _tensor(self._asset.data.soft_joint_pos_limits)[:, self._joint_ids]
        target.clamp_(limits[..., 0], limits[..., 1])
        self._asset.set_joint_position_target_index(target=target, joint_ids=self._joint_ids)

    def seed_joint_target(self, env_ids, joint_target: torch.Tensor) -> None:
        """Clear stale commands after a generated state is written."""
        del joint_target
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0


@configclass
class SoftLimitRelativeGripperActionCfg(RelativeJointPositionActionCfg):
    """Configuration for :class:`SoftLimitRelativeGripperAction`."""

    class_type: type[SoftLimitRelativeGripperAction] = SoftLimitRelativeGripperAction
