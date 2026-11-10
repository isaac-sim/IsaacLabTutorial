# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teacher-controlled physical replay for the pinned native Duck SuperMimic learner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from rsl_rl.algorithms import SuperMimic
    from tensordict import TensorDict


def qualified_teacher_receipt(checkpoint: Path, receipt_path: Path) -> dict:
    """Require a matching SAC policy with complete response qualification, independently of export parity."""
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("algorithm") != "sac"
        or receipt.get("valid") is not True
        or receipt.get("protocol_id") != "flat-velocity-v1"
        or receipt.get("conditions", {}).get("task") != "MicroDuckTutorial-Velocity-Flat"
    ):
        raise ValueError("Choose a valid Duck SAC teacher receipt.")
    metrics = receipt.get("metrics", {})
    if (
        receipt.get("qualification", {}).get("passed") is not True
        or metrics.get("fell") is not False
        or metrics.get("coverage") != 1.0
        or receipt.get("horizon_s") != 20.0
    ):
        raise ValueError("The teacher must qualify on the complete 20-second command-response protocol.")
    if receipt.get("command_delay_ms") != 0 or receipt.get("observation_delay_ms") != 0:
        raise ValueError("Teacher initialization currently requires nominal undelayed conditions.")
    if receipt.get("checkpoint_sha256") != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
        raise ValueError("Teacher qualification does not match the selected checkpoint weights.")
    return receipt


@torch.no_grad()
def prepare_teacher_transition(
    algorithm: SuperMimic,
    observation: TensorDict,
    action: torch.Tensor,
    previous_actions: torch.Tensor,
) -> None:
    """Capture a teacher control before stepping, then use native ``process_env_step`` afterward.

    The pinned learner exposes no external-action collector. Isolate its three pending-transition
    fields here, preserving the same observation clone and real action history as native ``act``.
    The native successor path retains every-step pre-reset states, labels and reset handling.

    Args:
        algorithm: Native SuperMimic instance without an unprocessed transition.
        observation: Physical observation immediately before the applied teacher control.
        action: Actual teacher control that the caller will pass to the simulator.
        previous_actions: Actual oldest-first controls captured before advancing actor history.
    """
    if algorithm._last_obs is not None or algorithm._last_actions is not None:
        raise RuntimeError("Process the pending physical transition before recording another teacher control.")
    if action.shape != algorithm.actor.previous_actions[:, -1].shape:
        raise ValueError("A teacher control must contain one raw action per environment and servo.")
    if previous_actions.shape != algorithm.actor.previous_actions.shape:
        raise ValueError("Teacher replay requires the native oldest-first action history.")
    if not torch.isfinite(action).all() or not torch.isfinite(previous_actions).all():
        raise ValueError("Teacher controls and action history must be finite.")
    keys = {key for name in ("state", "goal", "context") for key in algorithm.obs_groups[name]}
    algorithm._last_obs = observation.select(*keys).clone()
    algorithm._last_actions = action.clone()
    algorithm._last_previous_actions = previous_actions.clone()
    algorithm.actor.previous_actions[:, -1].copy_(action)
