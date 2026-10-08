"""The LEAPP actor must retain both image and proprioceptive inputs."""

import json

import pytest
import torch

leapp = pytest.importorskip("leapp")

from isaaclab_tutorial.utils.export_leapp import export_visual_actor  # noqa: E402


class _VisualActor(torch.nn.Module):
    def forward(self, proprioception: torch.Tensor, images: list[torch.Tensor]) -> torch.Tensor:
        return proprioception[:, :6] + images[0].mean(dim=(-2, -1))


def test_leapp_bundle_responds_to_new_images_and_joint_inputs(tmp_path):
    source = tmp_path / "source.pt"
    torch.jit.script(_VisualActor()).save(str(source))
    path = export_visual_actor(source, tmp_path / "bundle")
    runtime = leapp.InferenceManager(str(path))
    inputs = {"policy/proprioception": torch.zeros(1, 24), "policy/wrist_rgb": torch.zeros(1, 6, 48, 64)}
    zero = runtime.run_policy(inputs)["policy/action"].cpu().clone()
    inputs["policy/wrist_rgb"].fill_(0.25)
    image_changed = runtime.run_policy(inputs)["policy/action"].cpu().clone()
    torch.testing.assert_close(image_changed - zero, torch.full((1, 6), 0.25))
    inputs["policy/proprioception"].fill_(0.5)
    joints_changed = runtime.run_policy(inputs)["policy/action"].cpu()
    torch.testing.assert_close(joints_changed - image_changed, torch.full((1, 6), 0.5))
    contract = json.loads((path.parent / "contract.json").read_text())
    assert contract["max_absolute_error"] <= 1e-6
    assert contract["image_shape"] == [1, 6, 48, 64]
    assert contract["image_preprocessing"] == "RGB_uint8_divided_by_255"
    with pytest.raises(FileExistsError):
        export_visual_actor(source, path.parent)
