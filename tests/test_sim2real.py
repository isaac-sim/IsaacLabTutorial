"""Acceptance audits must retain the physics distribution used to train both policies."""

import pytest

from isaaclab_tutorial.tasks.place_vial.config.so101.sim2real_env_cfg import (
    SO101VialCameraSim2RealEnvCfg,
    SO101VialSim2RealEnvCfg,
)
from isaaclab_tutorial.tasks.place_vial.reset.curriculum import CANONICAL_START


@pytest.mark.parametrize("config", [SO101VialSim2RealEnvCfg, SO101VialCameraSim2RealEnvCfg])
def test_home_audit_retains_randomized_physics(config):
    cfg = config()
    material = cfg.events.vial_material.params.copy()
    mass = cfg.events.vial_mass.params.copy()
    cfg.play_mode()
    assert cfg.episode_length_s == 30.0
    assert cfg.events.vial_material.params == material
    assert cfg.events.vial_mass.params == mass
    assert cfg.events.reset_from_dataset.params["phase_weights"] == CANONICAL_START
    assert cfg.events.reset_from_dataset.params["home_position_noise"] == 0.02
    assert cfg.events.robot_joint_parameters.mode == "reset"
    assert cfg.events.robot_actuator_gains.params["asset_cfg"].joint_names != ["gripper"]


def test_state_and_student_share_physics():
    state, camera = SO101VialSim2RealEnvCfg(), SO101VialCameraSim2RealEnvCfg()
    camera_events = camera.events.to_dict()
    for visual_event in ("camera_mount", "robot_color", "desk_color", "rack_color"):
        assert camera_events.pop(visual_event)["mode"] == "reset"
    assert state.events.to_dict() == camera_events
    assert not hasattr(camera.observations, "policy")


def test_camera_geometry_randomization_survives_play_and_preserves_policy_resolution():
    cfg = SO101VialCameraSim2RealEnvCfg()
    projection = cfg.observations.wrist_rgb.image.params.copy()
    mount = cfg.events.camera_mount.params.copy()
    cfg.play_mode()
    assert cfg.observations.wrist_rgb.image.params == projection
    assert cfg.events.camera_mount.params == mount
    assert (cfg.scene.wrist_camera.height, cfg.scene.wrist_camera.width) == (60, 80)
    assert projection["projection_size"] == (48, 64)
    assert projection["focal_scale_range"] == (0.95, 1.05)
    assert projection["normalize_intensity"] is False
    assert projection["history_length"] == 2


def test_appearance_variant_preserves_baseline_and_randomizes_vial_parts():
    from isaaclab_tutorial.tasks.place_vial.config.so101.robust_env_cfg import SO101VialCameraAppearanceEnvCfg

    baseline = SO101VialCameraSim2RealEnvCfg()
    original = baseline.to_dict()
    cfg = SO101VialCameraAppearanceEnvCfg()
    assert baseline.to_dict() == original
    assert cfg.events.reset_from_dataset.params == baseline.events.reset_from_dataset.params
    assert cfg.scene.vial.spawn.usd_path.endswith("vial_labeled.usda")
    for part in ("body", "cap", "label"):
        event = getattr(cfg.events, f"vial_{part}_color")
        material = getattr(cfg.scene, event.params["materials"].name)
        assert material.channels == ("color",)
        assert event.mode == "reset"
    before = cfg.events.vial_body_color.params.copy()
    cfg.play_mode()
    assert cfg.events.vial_body_color.params == before


def test_label_is_visual_only_and_keeps_measured_colliders():
    from pathlib import Path

    from pxr import Usd, UsdPhysics

    from isaaclab_tutorial.assets import VIAL_USD

    original = Usd.Stage.Open(str(VIAL_USD))
    labeled = Usd.Stage.Open(str(Path(VIAL_USD).with_name("vial_labeled.usda")))

    def colliders(stage):
        return {
            str(prim.GetPath()): (prim.GetAttribute("radius").Get(), prim.GetAttribute("height").Get())
            for prim in stage.Traverse()
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        }

    assert colliders(original) == colliders(labeled)
    assert not labeled.GetPrimAtPath("/Vial/Label").HasAPI(UsdPhysics.CollisionAPI)
