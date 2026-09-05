"""Portable format for physics-validated task-horizon reset poses."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

FORMAT = "so101-vial-reset-dataset"
SCHEMA_VERSION = 2
PHASE_NAMES = (
    "approach",
    "pregrasp",
    "grasp",
    "lift",
    "reorient",
    "transport",
    "insert",
    "release",
)
STATE_SHAPES = {
    "joint_position": (6,),
    "joint_target": (6,),
    "vial_pose": (7,),
    "phase": (),
    "difficulty": (),
    "grasped": (),
    "lifted": (),
}
_FLOAT_FIELDS = ("joint_position", "joint_target", "vial_pose", "difficulty")
_BOOL_FIELDS = ("grasped", "lifted")
_INTEGER_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


def _validate_state_fields(states: Mapping[str, Any]) -> None:
    if set(states) != set(STATE_SHAPES):
        missing = sorted(set(STATE_SHAPES) - set(states), key=repr)
        extra = sorted(set(states) - set(STATE_SHAPES), key=repr)
        raise ValueError(f"Reset state fields do not match the schema (missing={missing}, extra={extra}).")
    for name, value in states.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Reset field {name!r} must be a tensor.")
    for name in _FLOAT_FIELDS:
        if not states[name].is_floating_point():
            raise ValueError(f"Reset field {name!r} must have a floating-point dtype.")
    if states["phase"].dtype not in _INTEGER_DTYPES:
        raise ValueError("Reset field 'phase' must have an integer dtype.")
    for name in _BOOL_FIELDS:
        if states[name].dtype != torch.bool:
            raise ValueError(f"Reset field {name!r} must have a boolean dtype.")


def _canonical_states(states: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Return detached CPU tensors with stable storage dtypes."""
    if not isinstance(states, Mapping):
        raise ValueError("Reset states must be a mapping.")
    _validate_state_fields(states)
    return {
        "joint_position": states["joint_position"].detach().cpu().to(torch.float32).contiguous(),
        "joint_target": states["joint_target"].detach().cpu().to(torch.float32).contiguous(),
        "vial_pose": states["vial_pose"].detach().cpu().to(torch.float32).contiguous(),
        "phase": states["phase"].detach().cpu().to(torch.int64).contiguous(),
        "difficulty": states["difficulty"].detach().cpu().to(torch.float32).contiguous(),
        "grasped": states["grasped"].detach().cpu().to(torch.bool).contiguous(),
        "lifted": states["lifted"].detach().cpu().to(torch.bool).contiguous(),
    }


def validate_reset_states(states: Mapping[str, Any]) -> int:
    """Validate reset tensors and return their common row count."""
    if not isinstance(states, Mapping):
        raise ValueError("Reset states must be a mapping.")
    _validate_state_fields(states)
    row_count = int(states["phase"].shape[0]) if states["phase"].ndim == 1 else 0
    if row_count == 0:
        raise ValueError("Reset dataset must contain at least one row.")
    for name, trailing_shape in STATE_SHAPES.items():
        expected = (row_count, *trailing_shape)
        if tuple(states[name].shape) != expected:
            raise ValueError(f"Reset field {name!r} must have shape {expected}, got {tuple(states[name].shape)}.")
    for name in _FLOAT_FIELDS:
        if not bool(torch.isfinite(states[name]).all()):
            raise ValueError(f"Reset field {name!r} contains non-finite values.")
    if bool(((states["phase"] < 0) | (states["phase"] >= len(PHASE_NAMES))).any()):
        raise ValueError("Reset phase IDs are outside the declared phase table.")
    if bool(((states["difficulty"] < 0.0) | (states["difficulty"] > 1.0)).any()):
        raise ValueError("Reset difficulty must lie in [0, 1].")
    quaternion_norm = torch.linalg.vector_norm(states["vial_pose"][:, 3:7], dim=-1)
    if not bool(torch.allclose(quaternion_norm, torch.ones_like(quaternion_norm), atol=2.0e-3, rtol=0.0)):
        raise ValueError("Reset vial quaternions must be normalized XYZW quaternions.")
    return row_count


def _content_digest(states: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in STATE_SHAPES:
        tensor = states[name]
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def content_digest(states: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 digest of the ordered state tensors."""
    return _content_digest(_canonical_states(states))


def save_reset_dataset(
    path: str | Path,
    states: Mapping[str, Any],
    *,
    generator: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    """Validate and save one reset artifact."""
    if not isinstance(generator, dict) or not isinstance(validation, dict):
        raise ValueError("Reset artifact generator and validation metadata must be dictionaries.")
    canonical = _canonical_states(states)
    row_count = validate_reset_states(canonical)
    artifact = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "phase_names": PHASE_NAMES,
        "states": canonical,
        "generator": generator,
        "validation": validation,
        "content_sha256": _content_digest(canonical),
        "row_count": row_count,
    }
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = output.stat().st_mode & 0o777 if output.exists() else 0o644
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", delete=False) as file:
            temporary = Path(file.name)
        torch.save(artifact, temporary)
        os.chmod(temporary, mode)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return artifact


def load_reset_dataset(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    """Load and validate one reset artifact."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"Reset dataset not found at {source}. Generate it with `uv run generate-so101-resets`."
        )
    artifact = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or artifact.get("format") != FORMAT:
        raise ValueError(f"{source} is not an {FORMAT!r} artifact.")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported reset schema {artifact.get('schema_version')}; expected {SCHEMA_VERSION}.")
    if tuple(artifact.get("phase_names", ())) != PHASE_NAMES:
        raise ValueError("Reset phase table does not match this task version.")
    for name in ("generator", "validation"):
        if not isinstance(artifact.get(name), dict):
            raise ValueError(f"Reset artifact {name!r} metadata must be a dictionary.")
    states = _canonical_states(artifact.get("states"))
    row_count = validate_reset_states(states)
    stored_row_count = artifact.get("row_count")
    if not isinstance(stored_row_count, int) or isinstance(stored_row_count, bool) or stored_row_count != row_count:
        raise ValueError("Reset artifact row_count does not match its state tensors.")
    expected_digest = _content_digest(states)
    if artifact.get("content_sha256") != expected_digest:
        raise ValueError("Reset artifact content digest does not match its state tensors.")
    artifact["states"] = {name: tensor.to(device) for name, tensor in states.items()}
    return artifact
