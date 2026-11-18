# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Resolve the Microduck recipe's schedules on GA's fast configuration path.

GA skips OmegaConf interpolation when all overrides are consumed by its preset
resolver. Resolve only this project's microduck_scale expressions after those
overrides, preserving the reference's rank-aware schedules.
"""

import re

from isaaclab_tasks.utils import hydra

_original_register = hydra.register_task
_SCALE = re.compile(r"\$\{microduck_scale:(\d+),\$\{env.ngpus\},\$\{env.nnodes\}\}")


def register_task(*args, **kwargs):
    env, agent, hydra_args = _original_register(*args, **kwargs)
    if not type(env).__module__.startswith("isaaclab_tutorial.tasks.velocity.config.microduck."):
        return env, agent, hydra_args
    from .reference.flat_env_cfg import _scale_multi_gpu

    def resolve(value):
        match = _SCALE.fullmatch(value) if isinstance(value, str) else None
        return _scale_multi_gpu(int(match[1]), env.ngpus, env.nnodes) if match else value

    for name in ("max_iterations", "save_interval"):
        if agent is not None and hasattr(agent, name):
            setattr(agent, name, resolve(getattr(agent, name)))
    if env.curriculum is not None:
        for term in vars(env.curriculum).values():
            parameters = getattr(term, "params", {}).get("modify_params", {})
            if "stages" in parameters:
                parameters["stages"] = [(resolve(step), value) for step, value in parameters["stages"]]
    return env, agent, hydra_args


def install_config_resolution():
    hydra.register_task = register_task
