"""Visual model used by SO-101 policy distillation."""

import torch.nn as nn
from isaaclab_tasks.core.lift.config.kuka_allegro.agents.models import SpatialSoftmaxCNNModel
from tensordict import TensorDict


class GeometrySpatialSoftmaxCNNModel(SpatialSoftmaxCNNModel):
    """Spatial-softmax policy with a training-only geometry prediction head."""

    def __init__(self, *args, geometry_dim: int = 9, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if geometry_dim <= 0:
            raise ValueError("geometry_dim must be positive.")
        self.geometry_head = nn.Sequential(
            nn.Linear(self._get_latent_dim(), 128),
            nn.ELU(),
            nn.Linear(128, geometry_dim),
        )

    def predict_geometry(self, obs: TensorDict):
        """Predict normalized gripper-frame task geometry from actor inputs."""
        return self.geometry_head(self.get_latent(obs))
