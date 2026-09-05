"""Episode-local task milestones with safe partial-reset semantics."""

from collections.abc import Sequence

import torch


class PlacementProgress:
    """Track three sticky physical milestones and stable placement success.

    Each milestone is a physical fact that is latched the first time it is observed in an episode:

    * ``grasped``: both jaws touch the vial while it is held off the mat,
    * ``lifted``: the vial's lowest point has cleared the top of the rack,
    * ``inserted``: the vial's tip is inside the target rack opening.

    ``success`` latches after the released vial rests upright inside the opening for ``stable_steps`` consecutive
    control steps. Success is a purely physical outcome; it does not depend on which milestones were observed.
    """

    def __init__(self, num_envs: int, device: str | torch.device, stable_steps: int = 10, grasp_steps: int = 2):
        self.stable_steps = stable_steps
        self.grasp_steps = grasp_steps
        self.grasped = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.lifted = torch.zeros_like(self.grasped)
        self.inserted = torch.zeros_like(self.grasped)
        self.success = torch.zeros_like(self.grasped)
        self.unsafe_rack_contact = torch.zeros_like(self.grasped)
        self.grasp_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.stable_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.time_to_success = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    def reset(self, env_ids: Sequence[int] | slice | torch.Tensor | None = None) -> None:
        """Reset only the requested environments."""
        ids = slice(None) if env_ids is None else env_ids
        self.grasped[ids] = False
        self.lifted[ids] = False
        self.inserted[ids] = False
        self.success[ids] = False
        self.unsafe_rack_contact[ids] = False
        self.grasp_count[ids] = 0
        self.stable_count[ids] = 0
        self.time_to_success[ids] = -1

    def update(
        self,
        holding: torch.Tensor,
        cleared: torch.Tensor,
        inserted: torch.Tensor,
        seated: torch.Tensor,
        step: torch.Tensor,
    ) -> torch.Tensor:
        """Latch milestones from the current physical state and return confirmed success."""
        self.grasp_count.copy_(torch.where(holding, self.grasp_count + 1, torch.zeros_like(self.grasp_count)))
        self.grasped |= self.grasp_count >= self.grasp_steps
        self.lifted |= cleared
        self.inserted |= inserted
        self.stable_count.copy_(torch.where(seated, self.stable_count + 1, torch.zeros_like(self.stable_count)))
        newly_successful = ~self.success & (self.stable_count >= self.stable_steps)
        self.time_to_success.copy_(torch.where(newly_successful, step, self.time_to_success))
        self.success |= newly_successful
        return self.success.clone()
