"""Export a complete monocular SO-101 DINO policy and validate LEAPP parity."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch

from .control import JOINTS, PROPRIOCEPTION, validate_contract
from .model import load_policy
from .retina import nominal_calibration


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


def export(checkpoint: Path, dino: str, output: Path, calibration=None, device="cuda:0"):
    if device != "cuda:0":
        raise ValueError("Export on cuda:0; select the physical GPU with CUDA_VISIBLE_DEVICES")
    if output.exists():
        raise FileExistsError(f"Use a new export directory: {output}")
    settings = json.loads(checkpoint.with_name("example.json").read_text())
    planar = settings.get("retinal_geometry") == "planar"
    if planar and calibration is not None:
        raise ValueError("Planar deployment uses an uncalibrated square crop; omit --calibration")
    calibrated = calibration is not None
    calibration = calibration or nominal_calibration(settings["camera"]["camera_hfov"])
    dense, original, settings = load_policy(checkpoint, dino, calibration, device)
    output.mkdir(parents=True)
    camera = settings["camera"]
    scale = settings.get("control", {}).get("action_scale")
    if scale is None:
        raise ValueError("Checkpoint must record control.action_scale")
    contract = dict(
        format="isaaclab_fovi.so101.leapp.v1",
        joint_order=list(JOINTS),
        joint_units="radians",
        proprioception=list(PROPRIOCEPTION),
        policy_hz=30,
        relative_target_hz=120,
        action_scale=scale,
        gaze_limit=camera["gaze_limit"],
        gaze_max_delta=camera["gaze_max_delta"],
        camera_calibration=calibration,
        calibrated_capture=calibrated,
        retinal_geometry="planar" if planar else "spherical",
        capture_preprocessing="center_square_resize_640" if planar else "calibrated_raster",
        retinal_resolution=128,
        camera_hfov=camera["camera_hfov"],
        cmf_a=camera["cmf_a"],
        instruction=settings["policy"]["instructions"][0],
        source_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_checkpoint=checkpoint.name,
        dino_checksum=settings["policy"]["dino_checksum"],
        versions={name: importlib.metadata.version(name) for name in ("torch", "fovi", "leapp")},
        rgb_layout="BHWC_uint8",
        retinal_sampling="bilinear_border_mask_round_uint8",
        initial_gaze="neutral",
        output="eight_unclipped_normalized_actions",
    )
    validate_contract(contract)
    generator = torch.Generator(device=device).manual_seed(71)
    h, w = calibration["image_size"]
    proprio = torch.zeros(1, 32, device=device)
    rgb = torch.randint(0, 256, (1, h, w, 3), dtype=torch.uint8, device=device, generator=generator)
    retina = dense.retina(rgb, proprio[:, 26:28])
    camera_runtime, camera_yaml = export_node(dense, dict(proprio=proprio, rgb=rgb), output / "camera")
    retinal_runtime, retinal_yaml = export_node(
        dense.policy, dict(proprio=proprio, retinal_rgb=retina), output / "retinal"
    )
    maximum = dict(camera=0.0, retinal=0.0, original_actor=0.0)
    # Repeated-inference validation matters: the profiling executor changes this
    # BF16 encoder's numerics after its first invocation in PyTorch 2.11.
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        for _ in range(8):
            proprio = torch.randn(1, 32, device=device, generator=generator)
            proprio[:, 26:28].clamp_(-camera["gaze_limit"], camera["gaze_limit"])
            rgb.random_(0, 256, generator=generator)
            retina = dense.retina(rgb, proprio[:, 26:28])
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
    parser.add_argument("--dino", required=True, help="Local Fovi bundle or pinned model URI")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Actual capture CameraModel JSON for spherical policies; omit for planar policies",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    export(
        args.checkpoint.resolve(),
        args.dino,
        args.output.resolve(),
        json.loads(args.calibration.read_text()) if args.calibration else None,
        args.device,
    )


if __name__ == "__main__":
    main()
