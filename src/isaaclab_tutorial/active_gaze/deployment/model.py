"""Restore the trained actor without starting Isaac Lab or loading a text model."""

import copy
import json
from pathlib import Path

import torch
from torch import nn

from isaaclab_tutorial.active_gaze import RUN_FILE
from isaaclab_tutorial.active_gaze.learning.dino import FrozenDino
from isaaclab_tutorial.active_gaze.learning.models import LanguageVisionActor

from .control import GAZE, PROPRIOCEPTION_SIZE
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
    """Sample the retina from the camera image at the proprioception's gaze, then run the policy."""

    def __init__(self, retina, policy, field_radius_deg: float):
        super().__init__()
        self.retina, self.policy = retina, policy
        self.field_radius_deg = field_radius_deg

    def forward(self, proprio: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        return self.policy(proprio, self.retina(rgb, proprio[:, GAZE] * self.field_radius_deg))


def load_policy(checkpoint: Path, camera_model: dict, device: str, dino: str | None = None):
    """Load trusted local training artifacts and validate their supported contract.

    ``camera_model`` describes the real camera as :class:`fovi.sensing.projection.CameraModel` fields. ``dino``
    overrides the run's DINO reference, for example with a local bundle of the same checkpoint.
    """
    run = json.loads(checkpoint.with_name(RUN_FILE).read_text())
    config = json.loads(checkpoint.with_name("agent.json").read_text())
    retina = run["retina"]
    if run["policy"] != "dino_language" or retina["resolution"] != 128:
        raise ValueError("Deployment supports the 128px DINO and language policy")
    if run["action_chunk_size"] != 1:
        raise ValueError("Deployment requires single-step actions")
    if config["obs_groups"]["actor"] != ["proprio", "vision", "instruction"]:
        raise ValueError("Unsupported actor observation contract")
    if config.get("clip_actions") != 1.0:
        raise ValueError("Unsupported action clipping")
    packed = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = packed["actor_state_dict"]
    if state["obs_normalizer._mean"].shape != (1, PROPRIOCEPTION_SIZE):
        raise ValueError(f"Expected {PROPRIOCEPTION_SIZE} proprioceptive values")
    options = dict(config["actor"])
    if options.pop("class_name") != "isaaclab_tutorial.active_gaze.learning.models:LanguageVisionActor":
        raise ValueError("Unsupported actor class")
    options["instruction_table_path"] = str(checkpoint.with_name("instructions.pt"))
    options["language_device"] = "cpu"
    obs = dict(
        proprio=torch.zeros(1, PROPRIOCEPTION_SIZE),
        vision=torch.zeros(1, 64, 384),
        instruction=torch.zeros(1, 1, dtype=torch.long),
    )
    actor = LanguageVisionActor(obs, config["obs_groups"], "actor", 8, **options).eval()
    actor.load_state_dict(state, strict=True)
    settings = run["dino"]
    encoder = FrozenDino(
        dino or settings["reference"], settings["checksum"], batch_size=1, num_layers=settings["num_layers"]
    )
    retinal = RetinalPolicy(encoder, FlatActor(actor)).to(device).eval()
    sampler = SoftwareRetina(camera_model, retina["fov_deg"], retina["cmf_a_deg"], retina["resolution"])
    dense = CameraPolicy(sampler, retinal, run["gaze"]["field_radius_deg"])
    return dense.to(device).eval(), actor.to(device).eval(), run
