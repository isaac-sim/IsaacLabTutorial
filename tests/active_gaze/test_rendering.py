"""RTX minimal selection preserves numbered-task physics and policy image inputs."""

from types import SimpleNamespace

import pytest
import torch

MODES = [
    ("rtpt", "rgb", False),
    ("minimal_diffuse_mdl", "simple_shading_diffuse_mdl", False),
    ("minimal_diffuse_mdl_shadows", "simple_shading_diffuse_mdl", True),
    ("minimal_full_mdl", "simple_shading_full_mdl", False),
    ("minimal_full_mdl_shadows", "simple_shading_full_mdl", True),
]


@pytest.mark.parametrize("mode, output, shadows", MODES)
@pytest.mark.parametrize("policy", ["cnn", "dino_language"])
def test_mode_reaches_policy_and_diagnostic_cameras(mode, output, shadows, policy):
    from isaaclab_tutorial.active_gaze.configuration import compose
    from isaaclab_tutorial.active_gaze.env_cfg import SceneRGB
    from isaaclab_tutorial.active_gaze.train import add_diagnostic_camera

    variant = "dino_language" if policy == "dino_language" else "numbered"
    baseline, _, _ = compose([f"presets={variant}"])
    cfg, _, _ = compose([f"presets={variant},{mode}"])
    add_diagnostic_camera(cfg)
    assert cfg.scene.eye.data_types == cfg.scene.diagnostic_camera.data_types == [output]
    assert cfg.scene.eye.renderer_cfg.enable_shadows is shadows
    assert cfg.scene.diagnostic_camera.renderer_cfg.enable_shadows is shadows
    assert cfg.sim.physics.to_dict() == baseline.sim.physics.to_dict()
    assert cfg.scene.eye.spawn.to_dict() == baseline.scene.eye.spawn.to_dict()
    assert cfg.scene.eye.cmf_a_deg == baseline.scene.eye.cmf_a_deg
    image = torch.full((2, cfg.scene.eye.foveated_res, cfg.scene.eye.foveated_res, 3), 127, dtype=torch.uint8)
    camera = SimpleNamespace(cfg=cfg.scene.eye, data=SimpleNamespace(output={output: SimpleNamespace(torch=image)}))
    env = SimpleNamespace(scene={"eye": camera}, num_envs=2, device="cpu")
    actual = SceneRGB(SimpleNamespace(params={}), env)(env)
    torch.testing.assert_close(actual, image.float().permute(0, 3, 1, 2) / 255)
    if policy == "dino_language":
        assert cfg.observations.vision.tokens.params["data_type"] == output


def test_dino_consumes_the_selected_output_and_receives_only_encoder_arguments(monkeypatch):
    from isaaclab_tutorial.active_gaze.learning import observations

    image = torch.full((2, 128, 128, 3), 127, dtype=torch.uint8)
    received = {}

    class Encoder:
        def __init__(self, **kwargs):
            received.update(kwargs)

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, value):
            assert value is image
            return value

    monkeypatch.setattr(observations, "FrozenDino", Encoder)
    output = "simple_shading_full_mdl"
    params = dict(reference="local", checksum=None, batch_size=8, num_layers=6, data_type=output, photometric=None)
    eye = SimpleNamespace(data=SimpleNamespace(output={output: SimpleNamespace(torch=image)}))
    env = SimpleNamespace(device="cpu", num_envs=2, scene={"eye": eye})
    term = observations.DinoTokens(SimpleNamespace(params=params), env)
    assert received == dict(reference="local", checksum=None, batch_size=8, num_layers=6)
    assert term(env, **params) is image


def test_minimal_requires_ovrtx():
    from isaaclab_tutorial.active_gaze.configuration import compose

    with pytest.raises(ValueError, match="unsupported types"):
        compose(["presets=minimal_full_mdl", "env.scene.eye.renderer_cfg=newton_warp"])


@pytest.mark.parametrize("mode, output, shadows", [MODES[0], MODES[-1]])
def test_numbered_products_request_denoising_without_aa(mode, output, shadows):
    from pxr import Usd

    from isaaclab.renderers import CameraRenderSpec
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXCameraRenderData
    from isaaclab_ov.renderers.ovrtx_usd import build_render_product_as_string

    from isaaclab_tutorial.active_gaze.configuration import compose

    # Exercise the configured camera at the USD authoring boundary without a CUDA engine.
    cfg, _, _ = compose([f"presets=numbered,{mode}"])
    spec = CameraRenderSpec(
        cfg=cfg.scene.eye,
        camera_prim_paths=("/World/Camera",),
        num_instances=1,
        view_count=1,
        device="cpu",
    )
    data = OVRTXCameraRenderData(spec, "TestCamera", list(spec.camera_prim_paths))
    layer = build_render_product_as_string(spec, data, device_id=0, enable_shadows=shadows)
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().ImportFromString(layer)
    product = stage.GetPrimAtPath("/TestCamera/RenderProduct")
    assert product.GetAttribute("omni:rtx:post:aa:op").Get() == "none"
    assert product.GetAttribute("omni:rtx:newDenoiser:enabled").Get() is True
    samples = product.GetAttribute("omni:rtx:rtpt:spp")
    assert bool(samples) is (mode == "rtpt")
    if samples:
        assert samples.Get() == 4


@pytest.mark.parametrize("kind", ["cnn", "dino", "varied_dino"])
def test_visual_observations_refresh_only_at_policy_boundaries(monkeypatch, kind):
    from isaaclab_tutorial.active_gaze.env_cfg import SceneRGB
    from isaaclab_tutorial.active_gaze.learning import observations
    from isaaclab_tutorial.active_gaze.sim2real import PHOTOMETRIC_RANGES

    reads = []
    encodes = []
    image = torch.full((2, 4, 4, 3), 127, dtype=torch.uint8)

    class Camera:
        cfg = SimpleNamespace(data_types=["rgb"])

        @property
        def data(self):
            reads.append(True)
            return SimpleNamespace(output={"rgb": SimpleNamespace(torch=image)})

    class Encoder:
        def __init__(self, **kwargs):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, image):
            encodes.append(True)
            return image.float().mean(dim=(1, 2))

    monkeypatch.setattr(observations, "FrozenDino", Encoder)
    env = SimpleNamespace(device="cpu", num_envs=2, scene={"eye": Camera()}, refresh_policy_vision=False)
    params = {} if kind == "cnn" else dict(reference="local", checksum=None, batch_size=2)
    if kind == "varied_dino":
        params["photometric"] = PHOTOMETRIC_RANGES
    term_type = SceneRGB if kind == "cnn" else observations.DinoTokens
    term = term_type(SimpleNamespace(params=params), env)
    first = term(env, **params).clone()  # Empty cache must populate even during a chunk.
    image.fill_(255)
    torch.testing.assert_close(term(env, **params), first)
    assert len(reads) == 1
    env.refresh_policy_vision = True
    assert not torch.equal(term(env, **params), first)
    assert len(reads) == 2
    assert len(encodes) == (0 if kind == "cnn" else 2)
