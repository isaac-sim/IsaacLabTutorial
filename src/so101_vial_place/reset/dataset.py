"""Portable format for physics-validated task-horizon reset poses."""

from __future__ import annotations

import hashlib
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


def _canonical_states(states: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return detached CPU tensors with stable storage dtypes."""
    return {
        "joint_position": states["joint_position"].detach().cpu().to(torch.float32).contiguous(),
        "joint_target": states["joint_target"].detach().cpu().to(torch.float32).contiguous(),
        "vial_pose": states["vial_pose"].detach().cpu().to(torch.float32).contiguous(),
        "phase": states["phase"].detach().cpu().to(torch.int64).contiguous(),
        "difficulty": states["difficulty"].detach().cpu().to(torch.float32).contiguous(),
        "grasped": states["grasped"].detach().cpu().to(torch.bool).contiguous(),
        "lifted": states["lifted"].detach().cpu().to(torch.bool).contiguous(),
    }


def validate_reset_states(states: dict[str, torch.Tensor]) -> int:
    """Validate reset tensors and return their common row count."""
    if set(states) != set(STATE_SHAPES):
        missing = sorted(set(STATE_SHAPES) - set(states))
        extra = sorted(set(states) - set(STATE_SHAPES))
        raise ValueError(f"Reset state fields do not match the schema (missing={missing}, extra={extra}).")
    row_count = int(states["phase"].shape[0]) if states["phase"].ndim == 1 else 0
    if row_count == 0:
        raise ValueError("Reset dataset must contain at least one row.")
    for name, trailing_shape in STATE_SHAPES.items():
        expected = (row_count, *trailing_shape)
        if tuple(states[name].shape) != expected:
            raise ValueError(f"Reset field {name!r} must have shape {expected}, got {tuple(states[name].shape)}.")
    for name in ("joint_position", "joint_target", "vial_pose", "difficulty"):
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


def content_digest(states: dict[str, torch.Tensor]) -> str:
    """Return a deterministic SHA-256 digest of the ordered state tensors."""
    canonical = _canonical_states(states)
    digest = hashlib.sha256()
    for name in STATE_SHAPES:
        tensor = canonical[name]
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_reset_dataset(
    path: str | Path,
    states: dict[str, torch.Tensor],
    *,
    generator: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    """Validate and save one reset artifact."""
    canonical = _canonical_states(states)
    row_count = validate_reset_states(canonical)
    artifact = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "phase_names": PHASE_NAMES,
        "states": canonical,
        "generator": generator,
        "validation": validation,
        "content_sha256": content_digest(canonical),
        "row_count": row_count,
    }
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    return artifact


def load_reset_dataset(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    """Load and validate one reset artifact."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"Reset dataset not found at {source}. Generate it with `uv run isaaclab generate_resets`."
        )
    artifact = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or artifact.get("format") != FORMAT:
        raise ValueError(f"{source} is not an {FORMAT!r} artifact.")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported reset schema {artifact.get('schema_version')}; expected {SCHEMA_VERSION}.")
    if tuple(artifact.get("phase_names", ())) != PHASE_NAMES:
        raise ValueError("Reset phase table does not match this task version.")
    states = _canonical_states(artifact["states"])
    row_count = validate_reset_states(states)
    if artifact.get("row_count") != row_count:
        raise ValueError("Reset artifact row_count does not match its state tensors.")
    expected_digest = content_digest(states)
    if artifact.get("content_sha256") != expected_digest:
        raise ValueError("Reset artifact content digest does not match its state tensors.")
    artifact["states"] = {name: tensor.to(device) for name, tensor in states.items()}
    return artifact
