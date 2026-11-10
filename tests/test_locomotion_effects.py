# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical interventions retain their clocks, sensor semantics and matched mechanical starts."""

import copy
from dataclasses import asdict
from importlib.util import find_spec

import pytest

from isaaclab_tutorial.locomotion.deployment import DUCK_JOINTS
from isaaclab_tutorial.locomotion.effects import EffectSettings, mechanical_state
from isaaclab_tutorial.locomotion.playground.effects import EffectControls


def test_effects_are_bounded_explicit_physical_settings():
    settings = EffectSettings(command_delay_ms=20, observation_delay_ms=40, noise_scale=1.0, voltage=6.5)
    manifest = settings.manifest()
    assert manifest["command_delay_physics_steps"] == 4
    assert manifest["observation_delay_samples"] == 2
    assert manifest["noise_bounds"]["joint_velocity_rad_s"] == 0.25
    assert manifest["noise_bounds"]["angular_velocity_rad_s"] == 0.03
    for kwargs in (
        {"command_delay_ms": 3},
        {"observation_delay_ms": 5},
        {"voltage": float("nan")},
        {"noise_scale": -1},
    ):
        with pytest.raises(ValueError):
            EffectSettings(**kwargs)


def test_mechanical_state_combines_named_servo_and_play_coordinates_without_hiding_them():
    names = list(reversed(DUCK_JOINTS))
    passive = [f"passive_{name}_backlash" for name in names]
    raw = {
        "joint_names": names + passive,
        "joint_pos_rad": [0.2] * 14 + [0.01] * 14,
        "joint_vel_rad_s": [1.0] * 14 + [-0.1] * 14,
    }
    before = copy.deepcopy(raw)
    result = mechanical_state(raw)
    assert result["joint_names"] == list(DUCK_JOINTS)
    assert result["joint_pos_rad"] == pytest.approx([0.21] * 14)
    assert result["joint_vel_rad_s"] == pytest.approx([0.9] * 14)
    assert raw == before
    raw["joint_names"][-1] = "unrecognized_hinge"
    with pytest.raises(ValueError, match="joint"):
        mechanical_state(raw)


def test_slider_requests_snapshot_new_settings_without_changing_displayed_experiment():
    controls = EffectControls(EffectSettings())
    controls.values["voltage"] = 6.5
    controls.values["backlash"] = True
    controls.submit()
    controls.values["voltage"] = 8.2
    assert controls.applied == EffectSettings()
    assert controls.pending == EffectSettings(voltage=6.5, backlash=True)
    controls.values["observation_delay_ms"] = 3
    with pytest.raises(ValueError):
        controls.submit()
    assert controls.pending == EffectSettings(voltage=6.5, backlash=True)


def test_native_panel_drawing_does_not_turn_float_rounding_into_a_physical_edit():
    imgui = pytest.importorskip("imgui_bundle.imgui")
    context = imgui.create_context()
    try:
        io = imgui.get_io()
        io.display_size = (1200, 1000)
        io.delta_time = 1 / 60
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        controls = EffectControls(EffectSettings())
        for _ in range(2):
            imgui.new_frame()
            imgui.begin("Native effect panel")
            controls.draw(imgui)
            imgui.end()
            imgui.render()
        assert controls.values == asdict(controls.applied)
        assert controls.pending is None
    finally:
        imgui.destroy_context(context)


@pytest.mark.skipif(find_spec("isaaclab_assets.robots.microduck") is None, reason="Requires pinned locomotion source")
def test_effect_configuration_keeps_nominal_physics_and_uses_real_backlash_mapping():
    from isaaclab.utils import to_dict

    from isaaclab_tasks.utils import resolve_task_config

    import isaaclab_tutorial.tasks  # noqa: F401
    from isaaclab_tutorial.locomotion.effects import apply_effects

    cfg, _ = resolve_task_config("MicroDuckTutorial-Velocity-Flat", "rsl_rl_cfg_entry_point", overrides=())
    original = to_dict(cfg)
    apply_effects(cfg, EffectSettings())
    assert to_dict(cfg) == original
    apply_effects(cfg, EffectSettings(backlash=True, noise_scale=1, voltage=6.5, encoder_bias_rad=0.01))
    assert cfg.scene.robot.spawn.usd_path.endswith("microduck_walk_backlash.usd")
    actuator = cfg.scene.robot.actuators["servos"]
    assert actuator.backlash_joint_template == "passive_{}_backlash"
    assert actuator.min_delay == actuator.max_delay == 0
    assert actuator.vin_range == (6.5, 6.5) and actuator.vin_drop_gain_range == (0, 0)
    assert cfg.observations.policy.enable_corruption
    assert cfg.observations.policy.joint_vel.noise.n_max == 0.25
    assert len(cfg.observations.policy.joint_pos.params["backlash_cfg"].joint_names) == 14
    assert cfg.rewards.action_rate_l2.weight == original["rewards"]["action_rate_l2"]["weight"]
    assert cfg.events.encoder_bias.params["bias_range"] == (0.01, 0.01)
    assert cfg.curriculum is None
    assert cfg.commands.head_pose.ranges == ((0.0, 0.0),) * 4
