# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end checks of Newton-native BAM on live MJWarp articulations."""

from dataclasses import fields

import numpy as np
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators.newton import write_group_parameter
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.cloner import CloneCfg, clone_plan_from_env_0, replicate
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.test.utils import DeviceScope, test_devices as available_devices
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonSolverCfg

from isaaclab_tutorial.tasks.velocity.config.microduck.bam import install
from isaaclab_tutorial.tasks.velocity.config.microduck.bam.bam_cfg import BamActuatorCfg, BamMotorCfg

pytestmark = [
    pytest.mark.integration,
    pytest.mark.kitless,
]

DT = 1.0 / 120.0
"""Physics timestep [s]."""

BAM_DATA_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Tests/BAM"
"""Hosted BAM fixtures: the pendulum USD, its recorded trajectory and the upstream fit."""

NUM_ENVS = 2
"""Environments simulated side by side, so a per-environment randomization is observable."""

VIN = 7.4
"""Supply voltage the actuator is configured with [V]."""

KP_FW = 200.0
"""Firmware proportional gain the actuator is configured with [-]."""

INITIAL_ANGLE = 0.3
"""Angle the arm is released from [rad]. The commanded target is always 0."""


def _make_sim_cfg(
    device: str, solver_cfg: NewtonSolverCfg | None = None, use_newton_actuators: bool = True
) -> SimulationCfg:
    """Build the simulation configuration, with this module's MJWarp solver unless *solver_cfg* is given."""
    if solver_cfg is None:
        solver_cfg = MJWarpSolverCfg(njmax=20, nconmax=20, ls_iterations=20, integrator="implicitfast", impratio=1)
    return SimulationCfg(
        dt=DT,
        device=device,
        use_newton_actuators=use_newton_actuators,
        physics=NewtonCfg(solver_cfg=solver_cfg, num_substeps=2, debug_mode=False),
    )


@pytest.fixture(scope="module")
def pendulum_usd() -> str:
    """Local path of the fixed-base BAM pendulum fixture."""
    return retrieve_file_path(f"{BAM_DATA_DIR}/bam_pendulum.usda")


@pytest.fixture
def native_sim(device):
    """Newton simulation context running the Newton-native actuator path."""
    install()
    with build_simulation_context(
        device=device, gravity_enabled=True, add_ground_plane=False, sim_cfg=_make_sim_cfg(device)
    ) as sim_ctx:
        sim_ctx._app_control_on_stop_handle = None  # noqa: SLF001
        yield sim_ctx


def _make_cfg(**overrides) -> BamActuatorCfg:
    """Use the identified fit recorded alongside the upstream reference outputs."""
    with np.load(retrieve_file_path(f"{BAM_DATA_DIR}/bam_xl330_m6_goldens.npz")) as data:
        params = {key.removeprefix("attr_"): data[key].item() for key in data.files if key.startswith("attr_")}
    params["resistance"] = params.pop("R")
    motor = BamMotorCfg(model="m6", **{f.name: params[f.name] for f in fields(BamMotorCfg) if f.name in params})
    kwargs = {"joint_names_expr": [".*"], "motor": motor, "vin": VIN, "kp_fw": KP_FW}
    kwargs.update(overrides)
    return BamActuatorCfg(**kwargs)


def _spawn_pendulums(sim, pendulum_usd: str, cfgs: dict[str, BamActuatorCfg]) -> list[Articulation]:
    """Spawn one BAM pendulum per entry of *cfgs* in each of :data:`NUM_ENVS` environments and reset."""
    for index in range(NUM_ENVS):
        sim_utils.create_prim(f"/World/Env_{index}", "Xform", translation=(index * 1.0, 0.0, 1.0))
    robots = [
        Articulation(
            ArticulationCfg(
                prim_path=f"/World/Env_[^/]*/{name}",
                spawn=sim_utils.UsdFileCfg(usd_path=pendulum_usd),
                init_state=ArticulationCfg.InitialStateCfg(joint_pos={"joint": INITIAL_ANGLE}),
                actuators={"servo": cfg},
            )
        )
        for name, cfg in cfgs.items()
    ]
    clone_plan_from_env_0(CloneCfg(clone_template="/World/Env_{}"), [robot.cfg for robot in robots], NUM_ENVS, 1.0)
    replicate(sim.get_clone_plan())
    sim.reset()
    return robots


@pytest.mark.parametrize(
    "device,decimation",
    [
        (device, decimation)
        for device in available_devices(DeviceScope.CUDA)
        for decimation in ((1, 2) if device.startswith("cuda") else (1,))
    ],
)
def test_native_pendulum_matches_recorded_trajectory(native_sim, device, pendulum_usd, decimation):
    """Replay command reversals, settling and per-environment friction changes.

    On CUDA the friction scales change after graph capture, so live parameter updates are
    exercised with both state-buffer parities.
    """
    (robot,) = _spawn_pendulums(native_sim, pendulum_usd, {"Robot": _make_cfg()})
    drive = robot.actuators["servo"].drive
    np.testing.assert_allclose(
        robot.data.joint_viscous_friction_coeff.torch.cpu().numpy(), _make_cfg().motor.friction_viscous, rtol=1e-6
    )
    native_sim.physics_manager.set_decimation(decimation)
    robot.write_joint_position_to_sim_index(position=torch.full_like(robot.data.joint_pos.torch, INITIAL_ANGLE))
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(robot.data.joint_vel.torch))
    robot.actuators.reset()
    with np.load(retrieve_file_path(f"{BAM_DATA_DIR}/bam_pendulum_trajectory.npz")) as golden:
        assert native_sim.get_physics_dt() == float(golden["dt"])
        traces = {name: [] for name in ("position", "velocity", "effort", "friction_budget")}
        for index in range(0, len(golden["target"]), decimation):
            if index == 16:
                # A joint-property resync must preserve the initialized passive damping.
                robot.write_joint_armature_to_sim_index(armature=robot.data.joint_armature.torch.clone())
            target = float(golden["target"][index])
            robot.actuators.target_command.set_position_index(value=torch.full_like(robot.data.joint_pos.torch, target))
            write_group_parameter(
                robot.actuators,
                "servo",
                "controller",
                "friction_scale",
                torch.as_tensor(golden["friction_scale"][index, :, None], device=robot.device),
            )
            robot.write_data_to_sim()
            native_sim.step()
            robot.update(native_sim.get_physics_dt() * decimation)
            traces["position"].append(robot.data.joint_pos.torch.cpu().numpy().copy())
            traces["velocity"].append(robot.data.joint_vel.torch.cpu().numpy().copy())
            traces["effort"].append(robot.actuators.applied_effort.torch.cpu().numpy().copy())
            traces["friction_budget"].append(drive.friction_budget.numpy().copy())
        for name, values in traces.items():
            # CPU and CUDA solver reductions differ slightly.
            np.testing.assert_allclose(
                np.stack(values), golden[name][decimation - 1 :: decimation], atol=2e-5, rtol=2e-4, err_msg=name
            )
