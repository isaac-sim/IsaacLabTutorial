# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""A plane, a MicroDuck, and the sensors that explain its contact with the ground."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tutorial.tasks.velocity.config.microduck.robot import MICRODUCK_CFG


# BEGIN scene
@configclass
class DuckSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=1.0),
        debug_vis=False,
    )
    robot = MICRODUCK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base(/.*/(ankle_left|ankle_right))?",
        history_length=3,
        track_air_time=True,
    )
    self_collision = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base/.*/ankle_left",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/Geometry/trunk_base/.*/ankle_right"],
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


# END scene
