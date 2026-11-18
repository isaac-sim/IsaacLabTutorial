"""Export a complete monocular SO-101 DINO policy and validate LEAPP parity."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch

from isaaclab_tutorial.active_gaze import RUN_FILE

from .control import FORMAT, GAZE, JOINTS, POLICY_HZ, PROPRIOCEPTION, PROPRIOCEPTION_SIZE, TARGET_HZ, validate_contract
from .model import load_policy


def export_node(model, inputs: dict, output: Path):
    """Annotate and compile the explicit policy model with LEAPP."""
    import leapp

    # The encoder already traces explicit BF16 casts. The JIT autocast pass can
    # otherwise insert a second mixed-precision conversion after its warmup run.
    torch._C._jit_set_autocast_mode(False)
    output.mkdir(parents=True)
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        traced = torch.jit.trace(model, tuple(inputs.values()), check_trace=False)
        model_path = output / "policy.pt"
        traced.save(str(model_path))
        expected = model(*inputs.values())
        leapp.start(str(output))
        try:
            annotated = leapp.annotate.input_tensors("policy", inputs)
            values = [annotated] if len(inputs) == 1 else annotated
            representative = expected
            for value in values:
                representative = representative + 0 * value.float().sum()
            leapp.annotate.output_tensors(
                "policy",
                {"action": representative},
                export_with=None,
                backend_params={"model_path": str(model_path), "copy_original_model": True},
            )
        finally:
            leapp.stop()
        leapp.compile_graph(visualize=False, validate=True)
    yaml = output / f"{output.name}.yaml"
    return leapp.InferenceManager(str(yaml)), yaml


def export(checkpoint: Path, output: Path, camera_model: dict | None = None, dino: str | None = None, device="cuda:0"):
    """Export a checkpoint for the real camera ``camera_model``; without one, for the simulated camera."""
    if device != "cuda:0":
        raise ValueError("Export on cuda:0; select the physical GPU with CUDA_VISIBLE_DEVICES")
    if output.exists():
        raise FileExistsError(f"Use a new export directory: {output}")
    run = json.loads(checkpoint.with_name(RUN_FILE).read_text())
    control = run["control"]
    if (control["policy_hz"], control["physics_hz"]) != (POLICY_HZ, TARGET_HZ):
        raise ValueError(f"Deployment runs the policy at {POLICY_HZ} Hz and motor targets at {TARGET_HZ} Hz")
    calibrated = camera_model is not None
    camera_model = camera_model or run["retina"]["camera_model"]
    dense, original, run = load_policy(checkpoint, camera_model, device, dino)
    output.mkdir(parents=True)
    retina = run["retina"]
    gaze = run["gaze"]
    contract = dict(
        format=FORMAT,
        joint_order=list(JOINTS),
        joint_units="radians",
        proprioception=list(PROPRIOCEPTION),
        policy_hz=POLICY_HZ,
        relative_target_hz=TARGET_HZ,
        action_scale=control["action_scale"],
        gaze=dict(step_size=gaze["step_size"], limit=gaze["limit"], field_radius_deg=gaze["field_radius_deg"]),
        camera_model=camera_model,
        calibrated_capture=calibrated,
        retina=dict(fov_deg=retina["fov_deg"], cmf_a_deg=retina["cmf_a_deg"], resolution=retina["resolution"]),
        instruction=run["instructions"][0],
        source_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_checkpoint=checkpoint.name,
        dino=dict(checksum=run["dino"]["checksum"], num_layers=run["dino"]["num_layers"]),
        versions={name: importlib.metadata.version(name) for name in ("torch", "fovi", "leapp")},
        rgb_layout="BHWC_uint8",
        retinal_sampling="bilinear_border_mask_round_uint8",
        initial_gaze="optical_axis",
        output="eight_unclipped_normalized_actions",
    )
    validate_contract(contract)
    generator = torch.Generator(device=device).manual_seed(71)
    h, w = camera_model["image_size"]
    proprio = torch.zeros(1, PROPRIOCEPTION_SIZE, device=device)
    rgb = torch.randint(0, 256, (1, h, w, 3), dtype=torch.uint8, device=device, generator=generator)
    retina = dense.retina(rgb, proprio[:, GAZE] * dense.field_radius_deg)
    camera_runtime, camera_yaml = export_node(dense, dict(proprio=proprio, rgb=rgb), output / "camera")
    retinal_runtime, retinal_yaml = export_node(
        dense.policy, dict(proprio=proprio, retinal_rgb=retina), output / "retinal"
    )
    maximum = dict(camera=0.0, retinal=0.0, original_actor=0.0)
    # Repeated-inference validation matters: the profiling executor changes this
    # BF16 encoder's numerics after its first invocation in PyTorch 2.11.
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        for _ in range(8):
            proprio = torch.randn(1, PROPRIOCEPTION_SIZE, device=device, generator=generator)
            # Keep the gaze inside the trained limit, as the controller does.
            norm = proprio[:, GAZE].norm(dim=-1, keepdim=True).clamp_min(gaze["limit"])
            proprio[:, GAZE] *= gaze["limit"] / norm
            rgb.random_(0, 256, generator=generator)
            retina = dense.retina(rgb, proprio[:, GAZE] * dense.field_radius_deg)
            tokens = dense.policy.encoder(retina)
            expected = original(
                dict(
                    proprio=proprio,
                    vision=tokens,
                    instruction=torch.zeros(1, 1, dtype=torch.long, device=device),
                )
            )
            candidates = dict(
                original_actor=dense.policy(proprio, retina),
                camera=camera_runtime.run_policy({"policy/proprio": proprio, "policy/rgb": rgb})["policy/action"],
                retinal=retinal_runtime.run_policy({"policy/proprio": proprio, "policy/retinal_rgb": retina})[
                    "policy/action"
                ],
            )
            for name, actual in candidates.items():
                torch.testing.assert_close(actual.to(expected), expected, atol=2e-4, rtol=2e-4, msg=name)
                maximum[name] = max(maximum[name], (actual.to(expected) - expected).abs().max().item())
    contract.update(
        camera_graph=str(camera_yaml.relative_to(output)),
        retinal_graph=str(retinal_yaml.relative_to(output)),
    )
    # Every executable and graph is covered; the contract itself is the root manifest.
    contract["files"] = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    (output / "contract.json").write_text(json.dumps(contract, indent=2))
    (output / "parity.json").write_text(json.dumps(dict(device=device, samples=8, max_action_error=maximum), indent=2))
    print(json.dumps(dict(bundle=str(output), max_action_error=maximum), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--camera_model",
        type=Path,
        help="The real camera's CameraModel fields as JSON. Required on hardware; defaults to the simulated camera.",
    )
    parser.add_argument("--dino", help="Local Fovi bundle of the run's DINO checkpoint; defaults to the run's.")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    export(
        args.checkpoint.resolve(),
        args.output.resolve(),
        json.loads(args.camera_model.read_text()) if args.camera_model else None,
        args.dino,
        args.device,
    )


if __name__ == "__main__":
    main()
