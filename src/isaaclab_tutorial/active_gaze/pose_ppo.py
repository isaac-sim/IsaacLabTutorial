"""Camera-pose supervision trains the actor representation without a reward."""

import math

from isaaclab_tutorial.active_gaze.learning.oracle_ppo import OraclePPO

from .camera_geometry import pose_loss


class ExtrinsicsPPO(OraclePPO):
    def __init__(self, *args, extrinsics_weight=0.05, **kwargs):
        if not math.isfinite(extrinsics_weight) or extrinsics_weight < 0:
            raise ValueError("extrinsics_weight must be finite and nonnegative")
        super().__init__(*args, **kwargs)
        self.extrinsics_weight = extrinsics_weight
        if self._raw_actor.extrinsics_head is None:
            raise ValueError("Extrinsics PPO requires the actor's auxiliary head")

    def auxiliary_loss(self, observations):
        loss, metrics = pose_loss(self._raw_actor.extrinsics_prediction, observations["extrinsics"])
        return self.extrinsics_weight * loss, dict(metrics, extrinsics=loss)
