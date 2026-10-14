"""Package the exported visual actor as a LEAPP bundle with explicit image and proprioception inputs.

Image preprocessing, frame history and joint-target updates are done by ``deploy.py``, not the bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import torch


class _FlatVisualActor(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, proprioception: torch.Tensor, wrist_rgb: torch.Tensor) -> torch.Tensor:
        return self.actor(proprioception, [wrist_rgb])


def export_visual_actor(
    model_path: Path,
    output: Path,
    action_scale: tuple[float, ...],
    history: int = 2,
) -> Path:
    """Bundle an RSL-RL visual TorchScript export and verify LEAPP runtime parity on CPU.

    Inputs are [1, 24] proprioception and [1, 3 * history, 48, 64] preprocessed RGB.
    Outputs are six normalized actions, before clipping and target conversion.
    """
    import leapp

    if history < 1:
        raise ValueError("history must be positive")
    if len(action_scale) != 6 or any(not math.isfinite(value) or value <= 0 for value in action_scale):
        raise ValueError("Expected six positive finite action scales matching the evaluated policy")
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty export directory: {output}")
    actor = torch.jit.load(str(model_path), map_location="cpu").eval()
    generator = torch.Generator().manual_seed(71)
    examples = [
        (torch.randn(1, 24, generator=generator), torch.rand(1, 3 * history, 48, 64, generator=generator))
        for _ in range(8)
    ]
    with torch.inference_mode(), tempfile.TemporaryDirectory() as temporary:
        flat = torch.jit.trace(_FlatVisualActor(actor).eval(), examples[0])
        flat_path = Path(temporary) / "visual_actor.pt"
        flat.save(str(flat_path))
        example_output = flat(*examples[0])
        if example_output.shape != (1, 6):
            raise ValueError(f"Expected six SO-101 actions, got {tuple(example_output.shape)}")
        leapp.start(str(output))
        try:
            proprioception, wrist_rgb = examples[0]
            traced_proprioception, traced_rgb = leapp.annotate.input_tensors(
                "policy", {"proprioception": proprioception, "wrist_rgb": wrist_rgb}
            )
            # Tie the output to both inputs so LEAPP records them; parity is checked against the actor below.
            representative = example_output + 0 * traced_proprioception[:, :6] + 0 * traced_rgb.sum()
            leapp.annotate.output_tensors(
                "policy",
                {"action": representative},
                export_with=None,
                backend_params={"model_path": str(flat_path), "copy_original_model": True},
            )
        finally:
            leapp.stop()
        leapp.compile_graph(visualize=False, validate=True)
        yaml_path = output / f"{output.name}.yaml"
        runtime = leapp.InferenceManager(str(yaml_path))
        max_error = 0.0
        for proprioception, wrist_rgb in examples:
            expected = actor(proprioception, [wrist_rgb])
            actual = runtime.run_policy({"policy/proprioception": proprioception, "policy/wrist_rgb": wrist_rgb})[
                "policy/action"
            ].cpu()
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
            max_error = max(max_error, (actual - expected).abs().max().item())
    (output / "contract.json").write_text(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "history_length": history,
                "history_order": "oldest_first_repeat_initial_frame",
                "image_shape": [1, 3 * history, 48, 64],
                "image_preprocessing": "RGB_uint8_divided_by_255",
                "proprioception": ["joint_pos", "joint_vel", "joint_target", "previous_clipped_action"],
                "joint_order": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
                "joint_units": "radians",
                "learned_normalization_in_model": True,
                "output": "normalized_action_requires_clipping_and_relative_target_conversion",
                "policy_hz": 30,
                "relative_target_hz": 120,
                "action_scale": list(action_scale),
                "parity_samples": len(examples),
                "max_absolute_error": max_error,
                "leapp_version": leapp.__version__,
            },
            indent=2,
        )
        + "\n"
    )
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="RSL-RL exported visual policy.pt, not a checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="New LEAPP bundle directory")
    parser.add_argument("--history", type=int, default=2)
    parser.add_argument(
        "--action-scale",
        type=float,
        nargs=6,
        required=True,
        help="Per-joint action scales of the trained task (actions.*.scale in the run's params/env.yaml)",
    )
    args = parser.parse_args()
    # Run the parity check on CPU; LEAPP otherwise picks CUDA when it is available.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(export_visual_actor(args.model, args.output, tuple(args.action_scale), args.history))


if __name__ == "__main__":
    main()
