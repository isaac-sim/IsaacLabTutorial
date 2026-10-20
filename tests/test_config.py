"""Configuration contracts for the state and camera tasks, with and without the ``sim2real`` preset."""

from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch
from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab_tasks.utils.hydra import resolve_presets

import isaaclab_tutorial.tasks  # noqa: F401  (registers the tasks)
from isaaclab_tutorial.assets import VIAL_LABELED_USD, VIAL_USD
from isaaclab_tutorial.tasks.place_vial.config.so101.agents.rsl_rl_distillation_cfg import (
    BoundedTeacherDistillation,
    SO101CameraDistillationRunnerCfg,
)
from isaaclab_tutorial.tasks.place_vial.config.so101.agents.rsl_rl_ppo_cfg import SO101StatePPORunnerCfg
from isaaclab_tutorial.tasks.place_vial.config.so101.camera_env_cfg import (
    CameraSim2RealEventsCfg,
    SO101VialCameraEnvCfg,
)
from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import (
    BASE_HEIGHT_CORRECTION,
    PREGRASP_GRIPPER_POSITION,
    EventsCfg,
    Sim2RealEventsCfg,
    SO101VialEnvCfg,
    _remove_non_gripper_colliders,
)
from isaaclab_tutorial.tasks.place_vial.reset.curriculum import ALL_PHASES, CANONICAL_START
from isaaclab_tutorial.utils import evaluation


def _state(*presets):
    return resolve_presets(SO101VialEnvCfg(), ["newton_mjwarp", *presets])


def _camera(*presets):
    return resolve_presets(SO101VialCameraEnvCfg(), ["newton_mjwarp", "newton_renderer", *presets])


def test_tasks_are_registered_with_rsl_rl_entry_points():
    ids = sorted(task for task in gym.registry if task.startswith("IsaacTutorial-"))
    assert ids == [
        "IsaacTutorial-Inspect-SO101",
        "IsaacTutorial-Place-Vial-SO101",
        "IsaacTutorial-Place-Vial-SO101-Camera",
    ]
    inspection, state, camera = (gym.spec(task).kwargs for task in ids)
    assert inspection["env_cfg_entry_point"].endswith("inspection_env_cfg:SO101InspectionEnvCfg")
    assert "rsl_rl_cfg_entry_point" not in inspection
    assert state["env_cfg_entry_point"].endswith("env_cfg:SO101VialEnvCfg")
    assert state["rsl_rl_cfg_entry_point"].endswith("rsl_rl_ppo_cfg:SO101StatePPORunnerCfg")
    assert camera["env_cfg_entry_point"].endswith("camera_env_cfg:SO101VialCameraEnvCfg")
    assert camera["rsl_rl_cfg_entry_point"].endswith("rsl_rl_distillation_cfg:SO101CameraDistillationRunnerCfg")


def test_nominal_state_task():
    cfg = _state()
    assert cfg.scene.num_envs == 4096
    assert cfg.decimation == 4
    assert cfg.sim.dt == pytest.approx(1.0 / 120.0)
    assert cfg.episode_length_s == 20.0
    assert type(cfg.events) is EventsCfg
    assert cfg.events.vial_material.mode == "startup"
    assert cfg.events.reset_from_dataset.params["phase_weights"] == ALL_PHASES
    assert cfg.actions.arm_action.scale == pytest.approx(0.033)
    assert cfg.actions.gripper_action.scale == pytest.approx(0.02)
    assert cfg.scene.robot.init_state.pos == (-0.05, 0.0, 0.0)
    assert cfg.scene.robot.init_state.joint_pos["gripper"] == pytest.approx(PREGRASP_GRIPPER_POSITION)
    assert cfg.scene.robot.spawn.shoulder_response is None
    assert cfg.scene.robot.spawn.hide_camera_housing is False
    assert cfg.scene.vial.spawn.usd_path == str(VIAL_USD)
    assert cfg.scene.vial.spawn.randomize_diameters is False
    for name in ("support", "fixed_jaw_support", "moving_jaw_support"):
        assert getattr(cfg.scene, name) is None
    assert cfg.rewards.jaw_support_force is None
    assert cfg.rewards.approach_progress.params["height_offset"] == 0.0
    assert cfg.rewards.joint_limits.weight < 0


def test_sim2real_state_task():
    cfg = _state("sim2real")
    assert cfg.episode_length_s == 30.0
    assert type(cfg.events) is Sim2RealEventsCfg
    assert cfg.events.vial_material.mode == "reset"
    assert cfg.events.vial_mass.mode == "reset"
    # The dataset's held rows were generated with the base 5 mm lower, so sim2real trains from home only.
    assert cfg.events.reset_from_dataset.params["phase_weights"] == CANONICAL_START
    assert cfg.actions.arm_action.scale["shoulder_lift"] == pytest.approx(0.12)
    assert cfg.actions.arm_action.scale["elbow_flex"] == pytest.approx(0.07)
    assert cfg.scene.robot.init_state.pos == (-0.05, 0.0, BASE_HEIGHT_CORRECTION)
    assert cfg.scene.robot.spawn.shoulder_response == (0.22, 0.1, 0.4)
    assert cfg.scene.vial.spawn.randomize_diameters is True
    assert cfg.scene.support is not None
    assert cfg.scene.fixed_jaw_support.filter_prim_paths_expr == ["{ENV_REGEX_NS}/Desk", "{ENV_REGEX_NS}/Support"]
    assert cfg.rewards.jaw_support_force is not None
    assert cfg.rewards.approach_progress.params["height_offset"] == 0.008


def test_camera_task_inherits_state_task_and_adds_the_wrist_camera():
    nominal = _camera()
    assert nominal.scene.num_envs == 1024
    assert type(nominal.events) is EventsCfg
    assert nominal.scene.robot.spawn.hide_camera_housing is False
    assert nominal.scene.vial.spawn.usd_path == str(VIAL_LABELED_USD)
    assert (nominal.scene.wrist_camera.width, nominal.scene.wrist_camera.height) == (80, 60)
    image = nominal.observations.wrist_rgb.image.params
    assert image["projection_size"] == (48, 64)
    assert image["history_length"] == 2
    assert image["focal_scale_range"] == (1.0, 1.0)
    assert image["blur_range"] == (0.0, 0.0)
    # The policy-resolution focal length is the nominal 64 px wrist camera.
    assert image["focal_length_pixels"] == pytest.approx(64 * 13.6 / 20.955)

    randomized = _camera("sim2real")
    assert type(randomized.events) is CameraSim2RealEventsCfg
    assert randomized.events.camera_mount is not None
    image = randomized.observations.wrist_rgb.image.params
    assert image["focal_scale_range"] == (0.95, 1.05)
    assert image["radial_distortion_range"] == (-0.04, 0.04)
    # The teacher and student see the same physics.
    state = _state("sim2real")
    assert randomized.scene.robot.init_state.pos == state.scene.robot.init_state.pos
    assert randomized.actions.arm_action.scale == state.actions.arm_action.scale
    for name in vars(state.events):
        assert getattr(randomized.events, name).to_dict() == getattr(state.events, name).to_dict()


def test_agent_observation_groups_exist_in_their_tasks():
    state, camera = _state(), _camera()
    for groups in SO101StatePPORunnerCfg().obs_groups.values():
        assert all(hasattr(state.observations, group) for group in groups)
    for groups in SO101CameraDistillationRunnerCfg().obs_groups.values():
        assert all(hasattr(camera.observations, group) for group in groups)
    # The teacher input in the camera task is exactly the state actor input.
    assert camera.observations.teacher_state.to_dict() == state.observations.policy.to_dict()


@pytest.mark.parametrize("presets", [(), ("sim2real",)])
def test_play_uses_home_starts_and_pins_vial_only_without_sim2real(monkeypatch, presets):
    monkeypatch.setattr(evaluation, "EXACT_EVALUATION_ACTIVE", True)
    cfg = _state(*presets)
    cfg.scene.num_envs = 1024
    material = cfg.events.vial_material.params["static_friction_range"]
    cfg.play_mode()
    assert cfg.scene.num_envs == 1024
    assert cfg.events.reset_from_dataset.params["sequential"] is True
    assert cfg.events.reset_from_dataset.params["phase_weights"] == CANONICAL_START
    expected = material if presets else (0.7, 0.7)
    assert cfg.events.vial_material.params["static_friction_range"] == expected


def test_reset_generator_uses_the_nominal_state_scene():
    from isaaclab_tutorial.tasks.place_vial.reset.generator import _generator_env_cfg

    cfg = _generator_env_cfg(8, "cpu", "newton_mjwarp")
    assert cfg.scene.num_envs == 8
    assert cfg.events.reset_from_dataset is None
    assert cfg.events.vial_mass.params["mass_distribution_params"] == (0.02, 0.02)
    assert cfg.rewards is None and cfg.terminations is None


def test_only_jaw_and_camera_mount_colliders_remain():
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    for link in ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"):
        body = UsdGeom.Xform.Define(stage, f"/Robot/{link}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        mesh = UsdGeom.Mesh.Define(stage, f"/Robot/{link}/collisions/mesh").GetPrim()
        UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr("convexHull")
    camera = UsdGeom.Cube.Define(stage, "/Robot/gripper/collisions/camera_mount/Cube").GetPrim()
    UsdPhysics.CollisionAPI.Apply(camera)
    _remove_non_gripper_colliders(root)
    kept = {str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)}
    assert kept == {
        "/Robot/gripper/collisions/mesh",
        "/Robot/moving_jaw_so101_v1/collisions/mesh",
        "/Robot/gripper/collisions/camera_mount/Cube",
    }


def test_rolling_resistance_only_enables_support_contacts():
    from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import configure_support_rolling_contacts

    builder = SimpleNamespace(
        shape_label=[
            "/env_0/Support/geometry",
            "/env_0/Desk/collision",
            "/env_0/Vial/body_collider",
            "/env_0/Robot/gripper/collision",
            "/env_0/Rack/collision",
            "/env_1/Support/geometry",
        ],
        shape_world=[0, 0, 0, 0, 0, 1],
        shape_material_mu_rolling=[0.05] * 6,
        shape_material_mu_torsional=[0.005] * 6,
        custom_attributes={"mujoco:condim": SimpleNamespace(values={i: 3 for i in range(6)})},
    )
    configure_support_rolling_contacts(builder, (0.0002, 0.002), (0.001, 0.005))
    assert list(builder.custom_attributes["mujoco:condim"].values.values()) == [6, 6, 3, 3, 3, 6]
    assert 0.0002 <= builder.shape_material_mu_rolling[0] <= 0.002
    assert builder.shape_material_mu_rolling[2:5] == [0.0, 0.0, 0.0]


def test_distillation_labels_are_clamped_to_the_executed_action():
    algorithm = object.__new__(BoundedTeacherDistillation)
    algorithm.student = lambda obs, stochastic_output: torch.full((obs.shape[0], 2), 0.25)
    algorithm.teacher = lambda obs: torch.tensor([[4.0, -3.0]]).expand(obs.shape[0], -1)
    algorithm.transition = SimpleNamespace()
    actions = algorithm.act(torch.zeros(3, 1))
    assert torch.equal(actions, torch.full((3, 2), 0.25))
    assert torch.equal(algorithm.transition.privileged_actions, torch.tensor([[1.0, -1.0]]).expand(3, -1))
