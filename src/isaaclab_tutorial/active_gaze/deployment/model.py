"""Restore the trained actor without starting Isaac Lab or loading a text model."""

import copy
import json
from pathlib import Path

import torch
from torch import nn

from isaaclab_tutorial.active_gaze.learning.dino import FrozenDino
from isaaclab_tutorial.active_gaze.learning.models import LanguageVisionActor

from .retina import SoftwareRetina


class FlatActor(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.normalizer = copy.deepcopy(actor.obs_normalizer)
        self.fusion = copy.deepcopy(actor.fusion)
        self.mlp = copy.deepcopy(actor.mlp)
        self.output = actor.distribution.as_deterministic_output_module()
        self.register_buffer("language", actor.language_features.clone())
        self.register_buffer("padding", actor.language_padding.clone())

    def forward(self, proprio: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:
        visual = self.fusion(vision, self.language.expand(len(vision), -1, -1), self.padding.expand(len(vision), -1))
        return self.output(self.mlp(torch.cat((self.normalizer(proprio), visual), -1)))


class RetinalPolicy(nn.Module):
    def __init__(self, encoder, actor):
        super().__init__()
        self.encoder, self.actor = encoder, actor

    def forward(self, proprio: torch.Tensor, retinal_rgb: torch.Tensor) -> torch.Tensor:
        return self.actor(proprio, self.encoder(retinal_rgb))


class CameraPolicy(nn.Module):
    def __init__(self, retina, policy):
        super().__init__()
        self.retina, self.policy = retina, policy

    def forward(self, proprio: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        return self.policy(proprio, self.retina(rgb, proprio[:, 26:28]))


def load_policy(checkpoint: Path, dino: str, calibration: dict, device: str):
    """Load trusted local training artifacts and validate their supported contract."""
    settings = json.loads(checkpoint.with_name("example.json").read_text())
    config = json.loads(checkpoint.with_name("agent.json").read_text())
    if settings["policy"]["name"] != "dino_language" or settings["resolution"] != 128:
        raise ValueError("This exporter supports the SO-101 128px DINO/language policy")
    if config["obs_groups"]["actor"] != ["proprio", "vision", "instruction"]:
        raise ValueError("Unsupported actor observation contract")
    if config["algorithm"].get("action_chunk_size", 1) != 1:
        raise ValueError("Hardware export currently requires single-step actions")
    if config.get("clip_actions") != 1.0:
        raise ValueError("Unsupported action clipping")
    packed = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = packed["actor_state_dict"]
    if state["obs_normalizer._mean"].shape != (1, 32):
        raise ValueError("Expected the SO-101 32-value proprioceptive contract")
    options = dict(config["actor"])
    if options.pop("class_name") != "isaaclab_tutorial.active_gaze.learning.models:LanguageVisionActor":
        raise ValueError("Unsupported actor class")
    options["instruction_table_path"] = str(checkpoint.with_name("instructions.pt"))
    options["language_device"] = "cpu"
    obs = dict(
        proprio=torch.zeros(1, 32),
        vision=torch.zeros(1, 64, 384),
        instruction=torch.zeros(1, 1, dtype=torch.long),
    )
    actor = LanguageVisionActor(obs, config["obs_groups"], "actor", 8, **options).eval()
    actor.load_state_dict(state, strict=True)
    encoder = FrozenDino(dino, settings["policy"]["dino_checksum"], batch_size=1)
    camera = settings["camera"]
    retinal = RetinalPolicy(encoder, FlatActor(actor)).to(device).eval()
    if settings.get("retinal_geometry") == "planar":
        from .planar import PlanarRetina

        retina = PlanarRetina(camera["camera_hfov"], camera["cmf_a"])
    else:
        retina = SoftwareRetina(calibration, camera["camera_hfov"], camera["cmf_a"])
    dense = CameraPolicy(retina, retinal)
    return dense.to(device).eval(), actor.to(device).eval(), settings
