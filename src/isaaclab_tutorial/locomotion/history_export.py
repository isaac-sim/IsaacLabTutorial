# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stream SuperMimic's sensor history through the standard Duck API-2 state tensors."""

from __future__ import annotations

import torch
from torch import nn


class DuckHistoryPolicy(nn.Module):
    """Keep the last sensor frames and a fresh flag; the network is not an LSTM.

    The standard runtime carries explicit ``h``/``c`` tensors and resets both to zero.
    ``h`` stores sensor history and a flag. ``c`` passes through unchanged. The first
    real frame fills the history, matching RSL-RL's reset padding exactly.
    """

    __constants__ = ("history_length", "state_width")

    def __init__(self, actor: nn.Module, history_length: int):
        super().__init__()
        if history_length < 1:
            raise ValueError("A Duck history actor needs at least one sensor frame.")
        self.actor = actor
        self.history_length = history_length
        self.state_width = (history_length - 1) * 48 + 1

    def forward(
        self, obs: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Consume one raw 61-value observation and return actions with the updated cache."""
        frame = obs[:, :48].unsqueeze(1)
        previous = h_in[0, :, :-1].reshape(1, self.history_length - 1, 48)
        history = torch.cat((previous, frame), dim=1)
        history = torch.where(h_in[0, :, -1:].unsqueeze(-1) > 0.5, history, frame.expand(1, self.history_length, 48))
        action = self.actor(history, obs[:, 48:])
        cache = torch.cat((history[:, 1:].flatten(1), torch.ones_like(obs[:, :1])), dim=-1).unsqueeze(0)
        return action, cache, c_in

    @torch.jit.ignore
    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the batch-one standard Duck input shapes for tracing and initialization."""
        return torch.zeros(1, 61), torch.zeros(1, 1, self.state_width), torch.zeros(1, 1, self.state_width)
