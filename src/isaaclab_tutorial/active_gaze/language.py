"""The same instruction is used for every cap and every SO-101 reset."""

INSTRUCTIONS = ("Place the vial in the slot with the matching number",)


def instruction_index(env):
    import torch

    return torch.zeros(env.num_envs, 1, device=env.device, dtype=torch.long)
