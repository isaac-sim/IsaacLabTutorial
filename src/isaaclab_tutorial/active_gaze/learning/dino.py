"""Frozen omni-foveated DINOv3 with the checkpoint's trained patch geometry."""

import json

import torch
from torch import nn

DEFAULT_CHECKPOINT = "wandb://nvidia/knnconv/5mgxp8y3#final_weights.pth"
DEFAULT_MD5 = "jsCUwCzwnDd1RRBPk8OVhQ=="


def load_backbone(reference: str, checksum: str | None) -> nn.Module:
    """Restore Fovi weights, including trained parametrizations, then fold LoRA."""
    from fovi.models import resolve_model_path
    from fovi.models.dinov3 import build_fovi_dinov3
    from fovi.models.loading import load_config
    from omegaconf import open_dict
    from torch.nn.utils import parametrize

    remote = reference.startswith("wandb:")
    path = resolve_model_path(reference, expected_checkpoint_md5=checksum if remote else None)
    if not path.is_dir():
        raise ValueError("Pass a Fovi model bundle directory, including its configuration and source.json.")
    if not remote and checksum is not None:
        # Fovi reserializes downloaded weights, so their file hash differs from
        # W&B's original. Preserve the bundle's original source identity and sizes,
        # following Fovi's cached-bundle validation contract.
        source = json.loads((path / "source.json").read_text())
        if (
            source["md5"] != checksum
            or source["state_dict_size"] != (path / "state_dict.pth").stat().st_size
            or source["config_size"] != (path / "resolved_config.yaml").stat().st_size
        ):
            raise ValueError(f"DINO bundle provenance or size mismatch: {path}")
    cfg, packed, key = load_config(path.name, True, path.parent, device="cpu")
    if cfg.model.arch != "fovi_dinov3" or cfg.saccades.mode != "warped_cartesian_as_grid":
        raise ValueError("Expected a Fovi DINOv3 warped_cartesian_as_grid checkpoint.")
    if (
        cfg.saccades.resize_size != 128
        or cfg.model.vit.patch_size != 16
        or cfg.saccades.get("radius_norm", 2.0) != 2.0
        or cfg.model.vit.position_coordinate_space != "cortical"
    ):
        raise ValueError("This example requires the 128px, patch16, circular cortical checkpoint.")
    with open_dict(cfg):
        cfg.model.output_mode = "patch_tokens"
        # The complete fine-tuned state below replaces every base-model weight.
        # Only the base architecture/processor configuration is needed here.
        cfg.pretrained_model.load_weights = False
        if (path / "base_model").is_dir():
            cfg.pretrained_model.path = str(path / "base_model")
    backbone = build_fovi_dinov3(cfg, device="cpu")
    prefix = "network.backbone."
    state = {
        name.removeprefix("module.").removeprefix(prefix): value
        for name, value in packed[key].items()
        if name.removeprefix("module.").startswith(prefix)
    }
    # Transformers moved encoder layers under model in newer releases.
    nested = any(name.startswith("model.layer.") for name in backbone.state_dict())
    state = {
        (
            ("model." + name)
            if nested and name.startswith("layer.")
            else name.removeprefix("model.")
            if not nested and name.startswith("model.layer.")
            else name
        ): value
        for name, value in state.items()
    }
    backbone.load_state_dict(state, strict=True)
    for child in backbone.modules():
        if parametrize.is_parametrized(child):
            for name in tuple(child.parametrizations):
                parametrize.remove_parametrizations(child, name, leave_parametrized=True)
    return backbone.requires_grad_(False).eval()


class FrozenDino(nn.Module):
    """Encode upright uint8 BHWC retinal images into unpooled patch tokens."""

    def __init__(
        self,
        reference: str = DEFAULT_CHECKPOINT,
        checksum: str | None = DEFAULT_MD5,
        batch_size: int = 128,
    ):
        super().__init__()
        if batch_size <= 0:
            raise ValueError("DINO batch size must be positive.")
        self.backbone = load_backbone(reference, checksum)
        self.batch_size = batch_size
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406))[None, :, None, None])
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225))[None, :, None, None])

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dtype != torch.uint8 or images.shape[1:] != (128, 128, 3):
            raise ValueError("DINO expects upright uint8 RGB images of shape (B,128,128,3).")
        outputs = []
        for chunk in images.split(self.batch_size):
            pixels = (chunk.permute(0, 3, 1, 2).float() / 255 - self.mean) / self.std
            with torch.autocast(images.device.type, dtype=torch.bfloat16, enabled=images.device.type == "cuda"):
                tokens = self.backbone(pixel_values=pixels)
            outputs.append(tokens.float())
        return torch.cat(outputs)
