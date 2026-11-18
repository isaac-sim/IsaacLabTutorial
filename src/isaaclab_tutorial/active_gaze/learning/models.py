"""Native RSL-RL feedforward actor with the CoRL language/vision readout."""

import torch
from rsl_rl.models import MLPModel

from .fusion import LanguageVisionFusion
from .language import instruction_table, trim_instruction_padding


class LanguageVisionActor(MLPModel):
    """Frozen retinal tokens and instruction tokens, trainable fusion and motor/gaze MLP.

    Only proprioception is empirically normalized. The instruction integer is a
    lookup index, never a numeric MLP feature. Privileged and oracle groups are
    excluded explicitly. Frozen DINO runs in the environment observation term.
    """

    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        *,
        instructions,
        instruction_table_path=None,
        language_device="cpu",
        fusion_dim=128,
        fusion_width=256,
        fusion_heads=4,
        fusion_blocks=2,
        spatial_channels=4,
        extrinsics_head=False,
        **kwargs,
    ):
        self.fusion_dim = fusion_dim
        self.instructions = tuple(instructions)
        saved = instruction_table(self.instructions, path=instruction_table_path, device=language_device)
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        features, padding = trim_instruction_padding(saved["features"], saved["padding"])
        self.register_buffer("language_features", features)
        self.register_buffer("language_padding", padding)
        self.fusion = LanguageVisionFusion(
            obs["vision"].shape[-1],
            self.language_features.shape[-1],
            obs["vision"].shape[-2],
            self.language_features.shape[-2],
            width=fusion_width,
            heads=fusion_heads,
            blocks=fusion_blocks,
            spatial_channels=spatial_channels,
            output_dim=fusion_dim,
        )
        self.extrinsics_head = None
        self.extrinsics_prediction = None
        if extrinsics_head:
            self.extrinsics_head = torch.nn.Sequential(
                torch.nn.Linear(self._get_latent_dim(), 128), torch.nn.ELU(), torch.nn.Linear(128, 9)
            )
            torch.nn.init.zeros_(self.extrinsics_head[-1].weight)
            torch.nn.init.zeros_(self.extrinsics_head[-1].bias)

    def _get_obs_dim(self, obs, obs_groups, obs_set):
        if set(obs_groups[obs_set]) != {"proprio", "vision", "instruction"}:
            raise ValueError("Actor inputs must be proprio, vision, instruction only.")
        if obs["vision"].ndim != 3 or obs["instruction"].shape[-1] != 1:
            raise ValueError("Expected spatial vision tokens and one instruction lookup index.")
        return ["proprio"], obs["proprio"].shape[-1]

    def _get_latent_dim(self):
        return self.obs_dim + self.fusion_dim

    def get_latent(self, obs, masks=None, hidden_state=None):
        indices = obs["instruction"].squeeze(-1).long()
        features = self.language_features.index_select(0, indices)
        padding = self.language_padding.index_select(0, indices)
        visual = self.fusion(obs["vision"], features, padding)
        latent = torch.cat((super().get_latent(obs), visual), dim=-1)
        if self.extrinsics_head is not None:
            self.extrinsics_prediction = self.extrinsics_head(latent)
        return latent

    def get_extra_state(self):
        return self.instructions

    def set_extra_state(self, state):
        if tuple(state) != self.instructions:
            raise ValueError("Checkpoint instruction vocabulary does not match the task.")

    def as_jit(self):
        raise NotImplementedError("Use the example play entry point for the full DINO/fusion pipeline.")

    def as_onnx(self, verbose=False):
        raise NotImplementedError("Use the example play entry point for the full DINO/fusion pipeline.")
