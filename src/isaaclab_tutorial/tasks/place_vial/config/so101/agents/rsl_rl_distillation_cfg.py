"""State-teacher to wrist-camera-student distillation configuration."""

import torch
from rsl_rl.algorithms import Distillation
from tensordict import TensorDict

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
)


class BoundedTeacherDistillation(Distillation):
    """RSL-RL distillation whose labels are the teacher's actions clipped to [-1, 1], as the env applies them."""

    def act(self, obs: TensorDict) -> torch.Tensor:
        actions = super().act(obs)
        self.transition.privileged_actions = self.transition.privileged_actions.clamp(-1.0, 1.0)
        return actions


# BEGIN lesson-06-distillation
@configclass
class SO101CameraDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    pass
# END lesson-06-distillation
