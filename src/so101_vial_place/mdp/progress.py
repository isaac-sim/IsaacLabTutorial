"""Episode-local placement history with safe partial-reset semantics."""

from collections.abc import Sequence

import torch


class PlacementProgress:
    """Track irreversible grasp/lift milestones and stable placement confirmation."""

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        stable_steps: int = 10,
        grasp_steps: int = 2,
    ):
        self.stable_steps = stable_steps
        self.grasp_steps = grasp_steps
        self.grasped = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.lifted = torch.zeros_like(self.grasped)
        self.grasp_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.stable_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.rack_hold_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.release_ready = torch.zeros_like(self.grasped)
        self.unsafe_rack_contact = torch.zeros_like(self.grasped)
        self.success = torch.zeros_like(self.grasped)
        self.time_to_success = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    def reset(self, env_ids: Sequence[int] | slice | torch.Tensor | None = None) -> None:
        """Reset only the requested environments."""
        ids = slice(None) if env_ids is None else env_ids
        self.grasped[ids] = False
        self.lifted[ids] = False
        self.grasp_count[ids] = 0
        self.stable_count[ids] = 0
        self.rack_hold_count[ids] = 0
        self.release_ready[ids] = False
        self.unsafe_rack_contact[ids] = False
        self.success[ids] = False
        self.time_to_success[ids] = -1

    def update(
        self,
        bilateral_contact: torch.Tensor,
        lifted_now: torch.Tensor,
        valid_released_placement: torch.Tensor,
        step: torch.Tensor,
        valid_grasped_placement: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance staged history and return confirmed success."""
        self.grasp_count.copy_(torch.where(bilateral_contact, self.grasp_count + 1, torch.zeros_like(self.grasp_count)))
        self.grasped |= self.grasp_count >= self.grasp_steps
        self.lifted |= self.grasped & lifted_now
        if valid_grasped_placement is not None:
            rack_candidate = self.lifted & valid_grasped_placement
            self.rack_hold_count.copy_(
                torch.where(rack_candidate, self.rack_hold_count + 1, torch.zeros_like(self.rack_hold_count))
            )
            self.release_ready |= self.rack_hold_count >= 1
        candidate = self.lifted & self.release_ready & valid_released_placement
        self.stable_count.copy_(torch.where(candidate, self.stable_count + 1, torch.zeros_like(self.stable_count)))
        newly_successful = (~self.success) & (self.stable_count >= self.stable_steps)
        self.time_to_success.copy_(torch.where(newly_successful, step, self.time_to_success))
        self.success |= newly_successful
        return self.success.clone()
