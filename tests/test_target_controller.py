"""Simulation and hardware use the same bounded setpoint recurrence."""

from types import SimpleNamespace

import numpy as np
import torch

from isaaclab_tutorial.tasks.place_vial.mdp.actions import RobustRelativeJointPositionAction
from isaaclab_tutorial.utils.deploy import TargetLimiter


def action():
    term = object.__new__(RobustRelativeJointPositionAction)
    term.cfg = SimpleNamespace(target_rate_deg_s=5.0, max_target_offset_deg=7.0)
    term._env = SimpleNamespace(physics_dt=1 / 120)
    term._joint_ids = [0, 1]
    term._asset = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=torch.zeros(2, 2), soft_joint_pos_limits=torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]] * 2)
        )
    )
    term._asset.set_joint_position_target_index = lambda **kwargs: None
    term._limited_target = torch.zeros(2, 2)
    term.envelope_violation = torch.zeros(2, dtype=torch.bool)
    term._raw_actions = torch.zeros(2, 2)
    term._processed_actions = torch.zeros(2, 2)
    return term


def test_simulation_matches_deployment_for_stall_tracking_and_reversal():
    sim = action()
    real = TargetLimiter(np.zeros(2), 5.0, 7.0)
    for i in range(1000):
        q = sim._asset.data.joint_pos
        # First world tracks imperfectly; second stays stalled throughout.
        q[0] += 0.1 * (sim._limited_target[0] - q[0])
        sim._processed_actions[:] = 0.12 if i < 500 else -0.07
        requested = (q[0] + sim._processed_actions[0]).clamp(-1, 1).numpy()
        expected = real.limit(requested, q[0].numpy(), 1 / 120)
        sim.apply_actions()
        np.testing.assert_allclose(sim._limited_target[0].numpy(), expected, atol=2e-6)
        assert not sim.envelope_violation.any()
        assert sim._limited_target[1].abs().max() <= np.radians(7) + 1e-6


def test_reset_seed_is_partial_and_envelope_failure_is_latched():
    sim = action()
    sim._asset.data.joint_pos[0] = 1.0
    sim.apply_actions()
    assert sim.envelope_violation.tolist() == [True, False]
    sim.seed_joint_target(torch.tensor([0]), torch.tensor([[0.1, 0.2]]))
    assert not sim.envelope_violation.any()
    torch.testing.assert_close(sim._limited_target, torch.tensor([[0.1, 0.2], [0.0, 0.0]]))
