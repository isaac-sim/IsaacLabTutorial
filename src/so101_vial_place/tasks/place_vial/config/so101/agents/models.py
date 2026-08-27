"""Small visual models specialized only at the representation boundary."""

import copy

import torch
import torch.nn as nn
from isaaclab_tasks.core.lift.config.kuka_allegro.agents.models import SpatialSoftmaxCNNModel
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict

POST_LIFT_GRIPPER_MAX = 0.25
POST_LIFT_SHOULDER_MIN = 0.75
POST_LIFT_WRIST_MAX = -1.60


def residual_post_lift_gate(proprioception: torch.Tensor) -> torch.Tensor:
    """Return the deployed residual-arm gate from the leading joint positions."""
    return (proprioception[..., 5:6] < POST_LIFT_GRIPPER_MAX) & (proprioception[..., 1:2] > POST_LIFT_SHOULDER_MIN)


def replacement_post_lift_gate(proprioception: torch.Tensor) -> torch.Tensor:
    """Return the deployed replacement-arm gate from the leading joint positions."""
    return residual_post_lift_gate(proprioception) & (proprioception[..., 3:4] < POST_LIFT_WRIST_MAX)


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

    def predict_geometry(self, obs: TensorDict) -> torch.Tensor:
        """Predict normalized gripper-frame task geometry from actor inputs."""
        return self.geometry_head(self.get_latent(obs))


class GeometryBottleneckSpatialSoftmaxCNNModel(SpatialSoftmaxCNNModel):
    """Act from predicted visual task geometry plus proprioception."""

    def __init__(self, *args, geometry_dim: int = 9, **kwargs) -> None:
        if geometry_dim <= 0:
            raise ValueError("geometry_dim must be positive.")
        self.geometry_dim = geometry_dim
        super().__init__(*args, **kwargs)
        self.geometry_head = nn.Sequential(
            nn.Linear(self.obs_dim + self.keypoint_dim, 128),
            nn.ELU(),
            nn.Linear(128, geometry_dim),
        )

    def _spatial_latent(self, obs: TensorDict) -> torch.Tensor:
        return SpatialSoftmaxCNNModel.get_latent(self, obs)

    def predict_geometry(self, obs: TensorDict) -> torch.Tensor:
        """Predict normalized geometry from spatial keypoints and proprioception."""
        return self.geometry_head(self._spatial_latent(obs))

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
        """Build the control bottleneck from proprioception and predicted geometry."""
        del masks, hidden_state
        proprioception = MLPModel.get_latent(self, obs)
        return torch.cat((proprioception, self.predict_geometry(obs)), dim=-1)

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self.geometry_dim

    def as_jit(self) -> nn.Module:
        """Return an export wrapper that preserves the geometry bottleneck."""
        return _TorchGeometryBottleneckModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return an ONNX wrapper that preserves the geometry bottleneck."""
        return _OnnxGeometryBottleneckModel(self, verbose)


class SplitGripperGeometryBottleneckCNNModel(GeometryBottleneckSpatialSoftmaxCNNModel):
    """Keep solved control fixed while a residual nonlinear head learns release."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gripper_mlp = copy.deepcopy(self.mlp)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output: bool = False):
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        latent = self.get_latent(obs, masks, hidden_state)
        arm_and_original_gripper = self.mlp(latent)
        gripper = arm_and_original_gripper[..., -1:] + self.gripper_mlp(latent)[..., -1:]
        output = torch.cat((arm_and_original_gripper[..., :-1], gripper), dim=-1)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(output)
        return output

    def as_jit(self) -> nn.Module:
        return _TorchSplitGripperGeometryBottleneckModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _OnnxSplitGripperGeometryBottleneckModel(self, verbose)


class ResidualSpatialSoftmaxCNNModel(SpatialSoftmaxCNNModel):
    """Learn arm corrections while structurally preserving the solved jaw policy."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.residual_mlp = copy.deepcopy(self.mlp)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output: bool = False):
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        latent = self.get_latent(obs, masks, hidden_state)
        original = self.mlp(latent)
        # Preserve acquisition exactly, then enable corrections on both the
        # canonical post-lift trajectory and bridge resets sampled from that
        # same trajectory.  Generated phase-five resets use a different
        # negative-shoulder configuration and must not be used with this gate.
        proprioception = obs["proprioception"]
        post_lift = residual_post_lift_gate(proprioception)
        arm = original[..., :-1] + post_lift.to(original.dtype) * self.residual_mlp(latent)[..., :-1]
        output = torch.cat((arm, original[..., -1:]), dim=-1)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(output)
        return output


class PostLiftSpatialSoftmaxCNNModel(ResidualSpatialSoftmaxCNNModel):
    """Switch from solved acquisition to an independently learned transport arm."""

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output: bool = False):
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        latent = self.get_latent(obs, masks, hidden_state)
        original = self.mlp(latent)
        proprioception = obs["proprioception"]
        post_lift = replacement_post_lift_gate(proprioception)
        transport = self.residual_mlp(latent)[..., :-1]
        arm = torch.where(post_lift, transport, original[..., :-1])
        output = torch.cat((arm, original[..., -1:]), dim=-1)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(output)
        return output


class GeometryAugmentedSpatialSoftmaxCNNModel(SpatialSoftmaxCNNModel):
    """Act from spatial keypoints, proprioception, and predicted task geometry."""

    def __init__(self, *args, geometry_dim: int = 9, **kwargs) -> None:
        if geometry_dim <= 0:
            raise ValueError("geometry_dim must be positive.")
        self.geometry_dim = geometry_dim
        super().__init__(*args, **kwargs)
        self.geometry_head = nn.Sequential(
            nn.Linear(self.obs_dim + self.keypoint_dim, 128),
            nn.ELU(),
            nn.Linear(128, geometry_dim),
        )

    def _spatial_latent(self, obs: TensorDict) -> torch.Tensor:
        return SpatialSoftmaxCNNModel.get_latent(self, obs)

    def predict_geometry(self, obs: TensorDict) -> torch.Tensor:
        return self.geometry_head(self._spatial_latent(obs))

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
        del masks, hidden_state
        spatial = self._spatial_latent(obs)
        return torch.cat((spatial, self.geometry_head(spatial)), dim=-1)

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self.keypoint_dim + self.geometry_dim

    def as_jit(self) -> nn.Module:
        return _TorchGeometryAugmentedModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _OnnxGeometryAugmentedModel(self, verbose)


class _TorchGeometryBottleneckModel(nn.Module):
    """Exportable deterministic geometry-bottleneck policy."""

    def __init__(self, model: GeometryBottleneckSpatialSoftmaxCNNModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(copy.deepcopy(model.cnns[group]), copy.deepcopy(model.softmaxes[group]))
                for group in model.obs_groups_2d
            ]
        )
        self.geometry_head = copy.deepcopy(model.geometry_head)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs_1d: torch.Tensor, obs_2d: list[torch.Tensor]) -> torch.Tensor:
        proprioception = self.obs_normalizer(obs_1d)
        spatial = [proprioception]
        for i, encoder in enumerate(self.encoders):
            spatial.append(encoder(obs_2d[i]))
        geometry = self.geometry_head(torch.cat(spatial, dim=-1))
        return self.deterministic_output(self.mlp(torch.cat((proprioception, geometry), dim=-1)))

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxGeometryBottleneckModel(_TorchGeometryBottleneckModel):
    """ONNX adapter for the geometry-bottleneck export wrapper."""

    def __init__(self, model: GeometryBottleneckSpatialSoftmaxCNNModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.obs_groups_2d = model.obs_groups_2d
        self.obs_dims_2d = model.obs_dims_2d
        self.obs_channels_2d = model.obs_channels_2d
        self.obs_dim_1d = model.obs_dim

    def forward(self, obs_1d: torch.Tensor, *obs_2d: torch.Tensor) -> torch.Tensor:
        return super().forward(obs_1d, list(obs_2d))

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        dummy_1d = torch.zeros(1, self.obs_dim_1d)
        dummy_2d = [
            torch.zeros(1, self.obs_channels_2d[i], *self.obs_dims_2d[i]) for i in range(len(self.obs_groups_2d))
        ]
        return (dummy_1d, *dummy_2d)

    @property
    def input_names(self) -> list[str]:
        return ["obs", *self.obs_groups_2d]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


class _TorchSplitGripperGeometryBottleneckModel(_TorchGeometryBottleneckModel):
    """Exportable split-head geometry-bottleneck policy."""

    def __init__(self, model: SplitGripperGeometryBottleneckCNNModel) -> None:
        super().__init__(model)
        self.gripper_mlp = copy.deepcopy(model.gripper_mlp)

    def forward(self, obs_1d: torch.Tensor, obs_2d: list[torch.Tensor]) -> torch.Tensor:
        proprioception = self.obs_normalizer(obs_1d)
        spatial = [proprioception]
        for i, encoder in enumerate(self.encoders):
            spatial.append(encoder(obs_2d[i]))
        geometry = self.geometry_head(torch.cat(spatial, dim=-1))
        latent = torch.cat((proprioception, geometry), dim=-1)
        original = self.mlp(latent)
        gripper = original[..., -1:] + self.gripper_mlp(latent)[..., -1:]
        return self.deterministic_output(torch.cat((original[..., :-1], gripper), dim=-1))


class _OnnxSplitGripperGeometryBottleneckModel(_TorchSplitGripperGeometryBottleneckModel):
    """ONNX adapter for the split-head bottleneck policy."""

    def __init__(self, model: SplitGripperGeometryBottleneckCNNModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.obs_groups_2d = model.obs_groups_2d
        self.obs_dims_2d = model.obs_dims_2d
        self.obs_channels_2d = model.obs_channels_2d
        self.obs_dim_1d = model.obs_dim

    def forward(self, obs_1d: torch.Tensor, *obs_2d: torch.Tensor) -> torch.Tensor:
        return super().forward(obs_1d, list(obs_2d))

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        dummy_1d = torch.zeros(1, self.obs_dim_1d)
        dummy_2d = [
            torch.zeros(1, self.obs_channels_2d[i], *self.obs_dims_2d[i]) for i in range(len(self.obs_groups_2d))
        ]
        return (dummy_1d, *dummy_2d)

    @property
    def input_names(self) -> list[str]:
        return ["obs", *self.obs_groups_2d]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


class _TorchGeometryAugmentedModel(_TorchGeometryBottleneckModel):
    """Exportable deterministic geometry-augmented policy."""

    def forward(self, obs_1d: torch.Tensor, obs_2d: list[torch.Tensor]) -> torch.Tensor:
        proprioception = self.obs_normalizer(obs_1d)
        spatial_parts = [proprioception]
        for i, encoder in enumerate(self.encoders):
            spatial_parts.append(encoder(obs_2d[i]))
        spatial = torch.cat(spatial_parts, dim=-1)
        geometry = self.geometry_head(spatial)
        return self.deterministic_output(self.mlp(torch.cat((spatial, geometry), dim=-1)))


class _OnnxGeometryAugmentedModel(_TorchGeometryAugmentedModel):
    """ONNX adapter for the geometry-augmented export wrapper."""

    def __init__(self, model: GeometryAugmentedSpatialSoftmaxCNNModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.obs_groups_2d = model.obs_groups_2d
        self.obs_dims_2d = model.obs_dims_2d
        self.obs_channels_2d = model.obs_channels_2d
        self.obs_dim_1d = model.obs_dim

    def forward(self, obs_1d: torch.Tensor, *obs_2d: torch.Tensor) -> torch.Tensor:
        return super().forward(obs_1d, list(obs_2d))

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        dummy_1d = torch.zeros(1, self.obs_dim_1d)
        dummy_2d = [
            torch.zeros(1, self.obs_channels_2d[i], *self.obs_dims_2d[i]) for i in range(len(self.obs_groups_2d))
        ]
        return (dummy_1d, *dummy_2d)

    @property
    def input_names(self) -> list[str]:
        return ["obs", *self.obs_groups_2d]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
