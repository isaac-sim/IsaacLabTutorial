"""Small, explicit checkpoint adapters used by the vision training recipe."""

from __future__ import annotations

import argparse
import copy
import math
from collections.abc import Sequence
from pathlib import Path

import torch


def _load(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {source} is not a dictionary.")
    return checkpoint


def _require_compatible(reference: dict, candidate: dict, label: str) -> None:
    """Reject a conversion when model parameter names or shapes differ."""
    if reference.keys() != candidate.keys():
        raise ValueError(f"{label} parameter names do not match the PPO model.")
    bad_shapes = [name for name in reference if reference[name].shape != candidate[name].shape]
    if bad_shapes:
        raise ValueError(f"{label} parameter shapes do not match the PPO model: {bad_shapes}")


def promote_distilled_student(
    distilled_path: str | Path,
    teacher_path: str | Path,
    ppo_template_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Create a PPO initialization checkpoint from native RSL-RL distillation.

    The student becomes the PPO actor and the state teacher's critic initializes
    the asymmetric critic.  A one-iteration camera PPO checkpoint supplies only
    the optimizer parameter-group layout; its stale moments are discarded.
    """
    distilled = _load(distilled_path)
    teacher = _load(teacher_path)
    template = _load(ppo_template_path)
    try:
        actor = distilled["student_state_dict"]
        critic = teacher["critic_state_dict"]
        template_actor = template["actor_state_dict"]
        template_critic = template["critic_state_dict"]
        optimizer = copy.deepcopy(template["optimizer_state_dict"])
    except KeyError as error:
        raise ValueError(f"Checkpoint is missing required field {error.args[0]!r}.") from error
    _require_compatible(template_actor, actor, "Student actor")
    _require_compatible(template_critic, critic, "Teacher critic")
    optimizer["state"] = {}
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": actor,
            "critic_state_dict": critic,
            "optimizer_state_dict": optimizer,
            "iter": 0,
            "infos": {
                "distilled_student": str(Path(distilled_path)),
                "teacher_critic": str(Path(teacher_path)),
            },
        },
        output,
    )
    return output


def recover_distillation_teacher(
    distilled_path: str | Path,
    ppo_template_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Recover the state actor embedded in a native distillation checkpoint.

    RSL-RL stores the complete frozen teacher inside every distillation save.
    A compatible PPO template supplies only the critic and optimizer layout;
    the optimizer state is cleared because it does not belong to this actor.
    """
    distilled = _load(distilled_path)
    template = copy.deepcopy(_load(ppo_template_path))
    try:
        teacher_actor = distilled["teacher_state_dict"]
        template_actor = template["actor_state_dict"]
        optimizer = template["optimizer_state_dict"]
    except KeyError as error:
        raise ValueError(f"Checkpoint is missing required field {error.args[0]!r}.") from error
    _require_compatible(template_actor, teacher_actor, "Distillation teacher")
    template["actor_state_dict"] = teacher_actor
    optimizer["state"] = {}
    template["iter"] = 0
    template["infos"] = {
        "recovered_distillation_teacher": str(Path(distilled_path)),
        "ppo_template": str(Path(ppo_template_path)),
    }
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(template, output)
    return output


def set_ppo_exploration_std(
    checkpoint_path: str | Path,
    output_path: str | Path,
    std: float | Sequence[float],
) -> Path:
    """Prepare a PPO checkpoint with explicit per-action exploration noise.

    The deterministic actor, critic, normalizers, and iteration are preserved.
    Only the Gaussian action standard deviation is changed, and stale optimizer
    moments are cleared so they cannot immediately undo the transition.
    """
    checkpoint = copy.deepcopy(_load(checkpoint_path))
    try:
        actor = checkpoint["actor_state_dict"]
        optimizer = checkpoint["optimizer_state_dict"]
    except KeyError as error:
        raise ValueError(f"Checkpoint is missing required field {error.args[0]!r}.") from error
    if "distribution.log_std_param" in actor:
        parameter_name = "distribution.log_std_param"
        stored_std = actor[parameter_name]
        logarithmic = True
    elif "distribution.std_param" in actor:
        parameter_name = "distribution.std_param"
        stored_std = actor[parameter_name]
        logarithmic = False
    else:
        raise ValueError("Checkpoint actor has no supported Gaussian standard-deviation parameter.")
    values = [float(std)] if isinstance(std, (int, float)) else [float(value) for value in std]
    if len(values) not in (1, stored_std.numel()):
        raise ValueError(f"std must contain one value or {stored_std.numel()} action values.")
    if any(not 0.0 < value <= 1.0 or not math.isfinite(value) for value in values):
        raise ValueError("std values must be finite and in (0, 1].")
    if len(values) == 1:
        values *= stored_std.numel()
    std_tensor = stored_std.new_tensor(values).reshape_as(stored_std)
    actor[parameter_name] = std_tensor.log() if logarithmic else std_tensor
    optimizer["state"] = {}
    infos = checkpoint.get("infos")
    if not isinstance(infos, dict):
        infos = {}
        checkpoint["infos"] = infos
    infos["exploration_std"] = values[0] if len(set(values)) == 1 else values
    infos["exploration_std_source"] = str(Path(checkpoint_path))
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return output


def add_geometry_head_from_template(
    checkpoint_path: str | Path,
    template_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Add a freshly initialized auxiliary head without changing the policy path."""
    checkpoint = _load(checkpoint_path)
    template = copy.deepcopy(_load(template_path))
    try:
        source_actor = checkpoint["actor_state_dict"]
        target_actor = template["actor_state_dict"]
        optimizer = template["optimizer_state_dict"]
    except KeyError as error:
        raise ValueError(f"Checkpoint is missing required field {error.args[0]!r}.") from error
    missing = [name for name in source_actor if name not in target_actor]
    bad_shapes = [
        name
        for name in source_actor
        if name in target_actor and source_actor[name].shape != target_actor[name].shape
    ]
    if missing or bad_shapes:
        raise ValueError(
            f"Template policy path is incompatible; missing={missing}, bad_shapes={bad_shapes}."
        )
    for name, value in source_actor.items():
        target_actor[name] = value
    template["critic_state_dict"] = checkpoint["critic_state_dict"]
    optimizer["state"] = {}
    template["iter"] = checkpoint.get("iter", 0)
    infos = template.get("infos")
    if not isinstance(infos, dict):
        infos = {}
        template["infos"] = infos
    infos["geometry_head_source"] = str(Path(template_path))
    infos["policy_source"] = str(Path(checkpoint_path))
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(template, output)
    return output


def augment_geometry_policy_from_template(
    checkpoint_path: str | Path,
    template_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Expose predicted geometry to a policy while preserving its initial actions."""
    checkpoint = _load(checkpoint_path)
    template = copy.deepcopy(_load(template_path))
    source_actor = checkpoint["actor_state_dict"]
    target_actor = template["actor_state_dict"]
    incompatible: list[str] = []
    for name, value in source_actor.items():
        if name not in target_actor:
            incompatible.append(name)
        elif target_actor[name].shape == value.shape:
            target_actor[name] = value
        elif (
            name == "mlp.0.weight"
            and target_actor[name].shape[0] == value.shape[0]
            and target_actor[name].shape[1] > value.shape[1]
        ):
            target_actor[name].zero_()
            target_actor[name][:, : value.shape[1]] = value
        else:
            incompatible.append(name)
    if incompatible:
        raise ValueError(f"Augmented template is incompatible at: {incompatible}.")
    template["critic_state_dict"] = checkpoint["critic_state_dict"]
    template["optimizer_state_dict"]["state"] = {}
    template["iter"] = checkpoint.get("iter", 0)
    infos = template.get("infos")
    if not isinstance(infos, dict):
        infos = {}
        template["infos"] = infos
    infos["augmented_policy_source"] = str(Path(checkpoint_path))
    infos["augmented_template_source"] = str(Path(template_path))
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(template, output)
    return output


def adapt_spatial_softmax_grid_from_template(
    checkpoint_path: str | Path,
    template_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Resize only deterministic spatial-softmax coordinate buffers for a new image size."""
    checkpoint = copy.deepcopy(_load(checkpoint_path))
    template = _load(template_path)
    actor = checkpoint["actor_state_dict"]
    template_actor = template["actor_state_dict"]
    grid_names = [name for name in actor if name.endswith((".pos_x", ".pos_y"))]
    if not grid_names:
        raise ValueError("Checkpoint actor has no spatial-softmax coordinate buffers.")
    for name in grid_names:
        if name not in template_actor:
            raise ValueError(f"Template is missing spatial-softmax buffer {name!r}.")
        actor[name] = template_actor[name]
    infos = checkpoint.get("infos")
    if not isinstance(infos, dict):
        infos = {}
        checkpoint["infos"] = infos
    infos["spatial_grid_source"] = str(Path(template_path))
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return output


def scale_geometry_policy_inputs(
    checkpoint_path: str | Path,
    output_path: str | Path,
    scale: float,
    geometry_dim: int = 9,
) -> Path:
    """Scale only an augmented policy's learned geometry-input connections."""
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive.")
    if geometry_dim <= 0:
        raise ValueError("geometry_dim must be positive.")
    checkpoint = copy.deepcopy(_load(checkpoint_path))
    try:
        actor = checkpoint["actor_state_dict"]
        weight = actor["mlp.0.weight"]
    except KeyError as error:
        raise ValueError(f"Checkpoint is missing required field {error.args[0]!r}.") from error
    if weight.ndim != 2 or weight.shape[1] <= geometry_dim:
        raise ValueError("Actor first layer is too small for the requested geometry inputs.")
    actor["mlp.0.weight"] = weight.clone()
    actor["mlp.0.weight"][:, -geometry_dim:] *= scale
    checkpoint["optimizer_state_dict"]["state"] = {}
    infos = checkpoint.get("infos")
    if not isinstance(infos, dict):
        infos = {}
        checkpoint["infos"] = infos
    infos["geometry_input_scale"] = scale
    infos["geometry_input_scale_source"] = str(Path(checkpoint_path))
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return output


def initialize_geometry_bottleneck_from_template(
    checkpoint_path: str | Path,
    template_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Reuse a scratch visual geometry encoder with a fresh bottleneck controller."""
    source = _load(checkpoint_path)
    template = copy.deepcopy(_load(template_path))
    source_actor = source["actor_state_dict"]
    target_actor = template["actor_state_dict"]
    prefixes = ("cnns.", "softmaxes.", "geometry_head.", "obs_normalizer.")
    selected = {name: value for name, value in source_actor.items() if name.startswith(prefixes)}
    missing = [name for name in selected if name not in target_actor]
    bad_shapes = [
        name for name, value in selected.items() if name in target_actor and target_actor[name].shape != value.shape
    ]
    if missing or bad_shapes:
        raise ValueError(f"Bottleneck template is incompatible; missing={missing}, bad_shapes={bad_shapes}.")
    for name, value in selected.items():
        target_actor[name] = value
    template["critic_state_dict"] = source["critic_state_dict"]
    template["optimizer_state_dict"]["state"] = {}
    template["iter"] = 0
    infos = template.get("infos")
    if not isinstance(infos, dict):
        infos = {}
        template["infos"] = infos
    infos["geometry_encoder_source"] = str(Path(checkpoint_path))
    infos["bottleneck_controller_source"] = str(Path(template_path))
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(template, output)
    return output


def interpolate_ppo_checkpoints(
    start_path: str | Path,
    end_path: str | Path,
    output_path: str | Path,
    alpha: float,
) -> Path:
    """Interpolate compatible PPO actor and critic weights along one training run."""
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1].")
    start = _load(start_path)
    end = _load(end_path)
    output_checkpoint = copy.deepcopy(end)
    for group in ("actor_state_dict", "critic_state_dict"):
        _require_compatible(start[group], end[group], group)
        mixed = {}
        for name, end_value in end[group].items():
            start_value = start[group][name]
            if torch.is_floating_point(end_value):
                mixed[name] = torch.lerp(start_value, end_value, alpha)
            else:
                mixed[name] = end_value.clone()
        output_checkpoint[group] = mixed
    output_checkpoint["optimizer_state_dict"]["state"] = {}
    output_checkpoint["iter"] = round(
        (1.0 - alpha) * int(start.get("iter", 0)) + alpha * int(end.get("iter", 0))
    )
    infos = output_checkpoint.get("infos")
    if not isinstance(infos, dict):
        infos = {}
        output_checkpoint["infos"] = infos
    infos["interpolation_start"] = str(Path(start_path))
    infos["interpolation_end"] = str(Path(end_path))
    infos["interpolation_alpha"] = alpha
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, output)
    return output


def interpolate_ppo_actor_prefix(
    start_path: str | Path,
    end_path: str | Path,
    output_path: str | Path,
    alpha: float,
    prefix: str,
) -> Path:
    """Interpolate or extrapolate one actor submodule while retaining the end elsewhere."""
    if not prefix:
        raise ValueError("prefix must be non-empty.")
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite.")
    start = _load(start_path)
    output_checkpoint = copy.deepcopy(_load(end_path))
    start_actor = start["actor_state_dict"]
    end_actor = output_checkpoint["actor_state_dict"]
    _require_compatible(start_actor, end_actor, "actor_state_dict")
    names = [name for name in end_actor if name.startswith(prefix)]
    if not names:
        raise ValueError(f"No actor parameters match prefix {prefix!r}.")
    for name in names:
        if torch.is_floating_point(end_actor[name]):
            end_actor[name] = torch.lerp(start_actor[name], end_actor[name], alpha)
    output_checkpoint["optimizer_state_dict"]["state"] = {}
    infos = output_checkpoint.get("infos")
    if not isinstance(infos, dict):
        infos = {}
        output_checkpoint["infos"] = infos
    infos["actor_interpolation_start"] = str(Path(start_path))
    infos["actor_interpolation_alpha"] = alpha
    infos["actor_interpolation_prefix"] = prefix
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, output)
    return output


def blend_ppo_output_rows(
    checkpoint_path: str | Path,
    source_path: str | Path,
    output_path: str | Path,
    rows: Sequence[int],
    alpha: float = 1.0,
    parts: str = "both",
) -> Path:
    """Blend or extrapolate selected final-action rows from a compatible checkpoint."""
    if not rows:
        raise ValueError("rows must be non-empty.")
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite.")
    if parts not in {"both", "weight", "bias"}:
        raise ValueError("parts must be both, weight, or bias.")
    checkpoint = copy.deepcopy(_load(checkpoint_path))
    source = _load(source_path)
    target_actor = checkpoint["actor_state_dict"]
    source_actor = source["actor_state_dict"]
    for name in ("mlp.6.weight", "mlp.6.bias"):
        if parts != "both" and not name.endswith(parts):
            continue
        target = target_actor[name].clone()
        candidate = source_actor[name]
        for row in rows:
            if row < 0 or row >= target.shape[0]:
                raise ValueError(f"Output row {row} is out of range for {name}.")
            target[row] = torch.lerp(target[row], candidate[row], alpha)
        target_actor[name] = target
    checkpoint["optimizer_state_dict"]["state"] = {}
    infos = checkpoint.get("infos")
    if not isinstance(infos, dict):
        infos = {}
        checkpoint["infos"] = infos
    infos["output_row_source"] = str(Path(source_path))
    infos["output_row_indices"] = list(rows)
    infos["output_row_alpha"] = alpha
    infos["output_row_parts"] = parts
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return output


def scale_ppo_output_rows(
    checkpoint_path: str | Path,
    output_path: str | Path,
    rows: Sequence[int],
    scale: float,
) -> Path:
    """Scale selected final-action rows for a diagnostic curriculum probe."""
    if not rows:
        raise ValueError("rows must be non-empty.")
    if not math.isfinite(scale):
        raise ValueError("scale must be finite.")
    checkpoint = copy.deepcopy(_load(checkpoint_path))
    actor = checkpoint["actor_state_dict"]
    for name in ("mlp.6.weight", "mlp.6.bias"):
        parameter = actor[name].clone()
        for row in rows:
            if row < 0 or row >= parameter.shape[0]:
                raise ValueError(f"Output row {row} is out of range for {name}.")
            parameter[row].mul_(scale)
        actor[name] = parameter
    checkpoint["optimizer_state_dict"]["state"] = {}
    infos = checkpoint.get("infos")
    if not isinstance(infos, dict):
        infos = {}
        checkpoint["infos"] = infos
    infos["scaled_output_row_indices"] = list(rows)
    infos["scaled_output_row_factor"] = scale
    infos["scaled_output_row_source"] = str(Path(checkpoint_path))
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return output


def add_split_gripper_head(checkpoint_path: str | Path, output_path: str | Path) -> Path:
    """Add a zero-output residual jaw head initialized from learned controller features."""
    checkpoint = copy.deepcopy(_load(checkpoint_path))
    actor = checkpoint["actor_state_dict"]
    controller = {name: value for name, value in actor.items() if name.startswith("mlp.")}
    if not controller:
        raise ValueError("Checkpoint actor has no controller MLP parameters.")
    if any(name.startswith("gripper_mlp.") for name in actor):
        raise ValueError("Checkpoint actor already has a split gripper head.")
    for name, value in controller.items():
        actor[name.replace("mlp.", "gripper_mlp.", 1)] = value.clone()
    actor["gripper_mlp.6.weight"].zero_()
    actor["gripper_mlp.6.bias"].zero_()
    checkpoint["optimizer_state_dict"]["state"] = {}
    infos = checkpoint.get("infos")
    if not isinstance(infos, dict):
        infos = {}
        checkpoint["infos"] = infos
    infos["split_gripper_source"] = str(Path(checkpoint_path))
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return output


def add_residual_head(checkpoint_path: str | Path, output_path: str | Path) -> Path:
    """Add a zero-output residual controller initialized from learned features."""
    checkpoint = copy.deepcopy(_load(checkpoint_path))
    actor = checkpoint["actor_state_dict"]
    controller = {name: value for name, value in actor.items() if name.startswith("mlp.")}
    if not controller:
        raise ValueError("Checkpoint actor has no controller MLP parameters.")
    if any(name.startswith("residual_mlp.") for name in actor):
        raise ValueError("Checkpoint actor already has a residual controller.")
    for name, value in controller.items():
        actor[name.replace("mlp.", "residual_mlp.", 1)] = value.clone()
    actor["residual_mlp.6.weight"].zero_()
    actor["residual_mlp.6.bias"].zero_()
    checkpoint["optimizer_state_dict"]["state"] = {}
    infos = checkpoint.get("infos")
    if not isinstance(infos, dict):
        infos = {}
        checkpoint["infos"] = infos
    infos["residual_policy_source"] = str(Path(checkpoint_path))
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return output


def geometry_input_scale_main(argv: list[str] | None = None) -> int:
    """CLI for scaling the predicted-geometry contribution to an augmented policy."""
    parser = argparse.ArgumentParser(description=scale_geometry_policy_inputs.__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", required=True, type=float)
    parser.add_argument("--geometry_dim", type=int, default=9)
    args = parser.parse_args(argv)
    output = scale_geometry_policy_inputs(
        args.checkpoint, args.output, args.scale, geometry_dim=args.geometry_dim
    )
    print(f"Wrote PPO checkpoint with scaled geometry inputs: {output}")
    return 0


def geometry_bottleneck_init_main(argv: list[str] | None = None) -> int:
    """CLI for initializing a visual geometry-bottleneck controller."""
    parser = argparse.ArgumentParser(description=initialize_geometry_bottleneck_from_template.__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = initialize_geometry_bottleneck_from_template(
        args.checkpoint, args.template, args.output
    )
    print(f"Wrote geometry-bottleneck initialization checkpoint: {output}")
    return 0


def interpolate_ppo_main(argv: list[str] | None = None) -> int:
    """CLI for interpolating two compatible PPO checkpoints."""
    parser = argparse.ArgumentParser(description=interpolate_ppo_checkpoints.__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", required=True, type=float)
    args = parser.parse_args(argv)
    output = interpolate_ppo_checkpoints(args.start, args.end, args.output, args.alpha)
    print(f"Wrote interpolated PPO checkpoint: {output}")
    return 0


def interpolate_ppo_actor_main(argv: list[str] | None = None) -> int:
    """CLI for interpolating one actor submodule between PPO checkpoints."""
    parser = argparse.ArgumentParser(description=interpolate_ppo_actor_prefix.__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args(argv)
    output = interpolate_ppo_actor_prefix(
        args.start, args.end, args.output, args.alpha, args.prefix
    )
    print(f"Wrote actor-submodule interpolated PPO checkpoint: {output}")
    return 0


def blend_ppo_output_rows_main(argv: list[str] | None = None) -> int:
    """CLI for blending selected PPO final-action rows."""
    parser = argparse.ArgumentParser(description=blend_ppo_output_rows.__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", required=True, type=int, nargs="+")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--parts", choices=("both", "weight", "bias"), default="both")
    args = parser.parse_args(argv)
    output = blend_ppo_output_rows(
        args.checkpoint, args.source, args.output, args.rows, args.alpha, args.parts
    )
    print(f"Wrote PPO checkpoint with blended output rows: {output}")
    return 0


def scale_ppo_output_rows_main(argv: list[str] | None = None) -> int:
    """CLI for scaling selected PPO final-action rows."""
    parser = argparse.ArgumentParser(description=scale_ppo_output_rows.__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", required=True, type=int, nargs="+")
    parser.add_argument("--scale", required=True, type=float)
    args = parser.parse_args(argv)
    output = scale_ppo_output_rows(args.checkpoint, args.output, args.rows, args.scale)
    print(f"Wrote PPO checkpoint with scaled output rows: {output}")
    return 0


def add_split_gripper_head_main(argv: list[str] | None = None) -> int:
    """CLI for initializing an independent jaw controller from a solved policy."""
    parser = argparse.ArgumentParser(description=add_split_gripper_head.__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = add_split_gripper_head(args.checkpoint, args.output)
    print(f"Wrote PPO checkpoint with an independent gripper head: {output}")
    return 0


def add_residual_head_main(argv: list[str] | None = None) -> int:
    """CLI for adding a behavior-preserving full-action residual controller."""
    parser = argparse.ArgumentParser(description=add_residual_head.__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = add_residual_head(args.checkpoint, args.output)
    print(f"Wrote PPO checkpoint with a residual controller: {output}")
    return 0


def promote_main(argv: list[str] | None = None) -> int:
    """CLI for converting a distilled wrist student into PPO initialization."""
    parser = argparse.ArgumentParser(description=promote_distilled_student.__doc__)
    parser.add_argument("--distilled", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--ppo_template", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = promote_distilled_student(args.distilled, args.teacher, args.ppo_template, args.output)
    print(f"Wrote PPO initialization checkpoint: {output}")
    return 0


def exploration_std_main(argv: list[str] | None = None) -> int:
    """CLI for preparing a PPO checkpoint with explicit exploration noise."""
    parser = argparse.ArgumentParser(description=set_ppo_exploration_std.__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--std", required=True, type=float, nargs="+")
    args = parser.parse_args(argv)
    output = set_ppo_exploration_std(args.checkpoint, args.output, args.std)
    print(f"Wrote PPO checkpoint with configured exploration: {output}")
    return 0
