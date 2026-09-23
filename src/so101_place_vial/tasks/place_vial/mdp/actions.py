"""Joint-space generator action and calibrated policy gripper action."""


import torch
from isaaclab.envs.mdp.actions import RelativeJointPositionAction, RelativeJointPositionActionCfg
from isaaclab.utils.configclass import configclass


def _tensor(value):
    """Return the torch view of an Isaac Lab proxy array."""
    return value.torch if hasattr(value, "torch") else value


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
