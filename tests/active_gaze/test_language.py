"""SO-101 language policy boundaries, gradients, and portable configuration."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict


def test_dino_early_exit_keeps_the_pretrained_layers():
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    from isaaclab_tutorial.active_gaze.learning.dino import keep_layers

    def config(layers):
        return DINOv3ViTConfig(
            hidden_size=32, intermediate_size=64, num_attention_heads=2, num_hidden_layers=layers, image_size=32
        )

    torch.manual_seed(0)
    full = DINOv3ViTModel(config(4)).eval()
    short = DINOv3ViTModel(config(2)).eval()
    short.load_state_dict(full.state_dict(), strict=False)
    pixels = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = short(pixel_values=pixels).last_hidden_state
        assert torch.equal(keep_layers(full, 2)(pixel_values=pixels).last_hidden_state, expected)
    with pytest.raises(ValueError, match="num_layers"):
        keep_layers(full, 3)


def test_language_configuration_keeps_numbered_task_and_critic():
    from isaaclab_tutorial.active_gaze.configuration import compose
    from isaaclab_tutorial.active_gaze.language import instruction_index

    cnn = compose(["presets=numbered", "env.scene.eye.cmf_a=0.0075"])[0]
    cfg = compose(["presets=dino_language", "env.scene.eye.cmf_a=0.0075"])[0]
    assert cfg.scene.eye.foveated_res == 128
    assert cfg.observations.scene_rgb is None
    assert cfg.observations.proprioception is None
    assert cfg.observations.gaze is None
    assert cfg.observations.critic.to_dict() == cnn.observations.critic.to_dict()
    assert cfg.terminations.to_dict() == cnn.terminations.to_dict()
    assert cfg.rewards.to_dict() == cnn.rewards.to_dict()
    assert cfg.events.to_dict() == cnn.events.to_dict()
    for key, value in cnn.observations.proprioception.to_dict().items():
        assert cfg.observations.proprio.to_dict()[key] == value
    env = SimpleNamespace(num_envs=4, device="cpu", vial_number=torch.arange(1, 5))
    assert torch.equal(instruction_index(env), torch.zeros(4, 1, dtype=torch.long))
    with pytest.raises(ValueError, match="128px"):
        compose(["presets=dino_language", "env.scene.eye.foveated_res=64"])
    assert cfg.goal_mode == "numbered"


@pytest.mark.parametrize("gaze_supervision", ["none", "grasp_gated"])
def test_language_actor_uses_vision_excludes_state_and_round_trips(tmp_path, gaze_supervision):
    from isaaclab_tutorial.active_gaze.learning.config import LanguageVisionActorCfg
    from isaaclab_tutorial.active_gaze.learning.language import COSMOS_MODEL, COSMOS_REVISION
    from isaaclab_tutorial.active_gaze.learning.models import LanguageVisionActor
    from isaaclab_tutorial.active_gaze.configuration import compose
    from isaaclab_tutorial.active_gaze.language import INSTRUCTIONS

    suffix = ",grasp_gated" if gaze_supervision == "grasp_gated" else ",no_gaze_supervision"
    cfg = compose([f"presets=dino_language{suffix}"])[1].to_dict()
    baseline = compose([f"presets=nearest{suffix}"])[1].to_dict()
    assert cfg["critic"] == baseline["critic"]
    assert cfg["obs_groups"]["critic"] == ["critic"]
    assert cfg["algorithm"].get("rnd_cfg") is None
    for key in ("hidden_dims", "fusion_dim", "fusion_width", "fusion_heads", "fusion_blocks", "spatial_channels"):
        assert cfg["actor"][key] == getattr(LanguageVisionActorCfg(), key)
    table_path = tmp_path / "instructions.pt"
    torch.save(
        dict(
            identity=dict(
                format=1,
                instructions=list(INSTRUCTIONS),
                checkpoint=COSMOS_MODEL,
                revision=COSMOS_REVISION,
                max_tokens=128,
            ),
            features=torch.randn(1, 128, 12).bfloat16(),
            padding=(torch.arange(128) >= 11)[None],
        ),
        table_path,
    )
    actor_cfg = cfg["actor"].copy()
    actor_cfg.pop("class_name")
    actor_cfg["instruction_table_path"] = str(table_path)
    obs = TensorDict(
        dict(
            proprio=torch.randn(4, 32),
            vision=torch.randn(4, 8, 16),
            instruction=torch.zeros(4, 1, dtype=torch.long),
            critic=torch.randn(4, 72),
            oracle=torch.randn(4, 3),
        ),
        batch_size=[4],
    )
    actor = LanguageVisionActor(obs, cfg["obs_groups"], "actor", 8, **actor_cfg)
    output = actor(obs)
    assert output.shape == (4, 8)
    changed = obs.clone()
    changed["critic"].fill_(float("nan"))
    changed["oracle"].fill_(float("nan"))
    torch.testing.assert_close(actor(changed), output)
    changed["vision"].zero_()
    assert not torch.allclose(actor(changed), output)
    output.square().sum().backward()
    assert actor.fusion.text_projection.weight.grad.abs().sum() > 0
    assert actor.fusion.vision_projection.weight.grad.abs().sum() > 0
    assert actor.mlp[-1].weight.grad[-2:].abs().sum() > 0
    assert not actor.language_features.requires_grad
    restored = LanguageVisionActor(obs, cfg["obs_groups"], "actor", 8, **actor_cfg)
    restored.load_state_dict(deepcopy(actor.state_dict()))
    torch.testing.assert_close(restored(obs), output)
