# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Blind rough walking using microduck_rl's gentle terrain recipe."""

import trimesh
from isaaclab_newton.sim.schemas import MujocoCollisionCfg

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from .walking import CurriculumCfg, WalkingSceneCfg, _schedule
from isaaclab_tutorial.tasks.velocity.mdp.curriculums import microduck_terrain_levels


class MicroDuckTerrainImporter(terrain_gen.TerrainImporter):
    """Author upstream's terrain contact response before Newton parses the scene."""

    def import_mesh(self, name: str, mesh: trimesh.Trimesh) -> None:
        """Import a terrain mesh and apply the softer MuJoCo contact response."""
        super().import_mesh(name, mesh)
        sim_utils.apply_collision_properties(
            f"{self.cfg.prim_path}/{name}/mesh",
            [MujocoCollisionCfg(solref=(0.04, 1.0), solimp=(0.85, 0.95, 0.001, 0.5, 2.0))],
        )


@configclass
class MicroDuckRoughSceneCfg(WalkingSceneCfg):
    """Walking scene with two terrain rays per foot, excluded from actor observations."""

    left_foot_height = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base/yaw2roll/hip_l/upper_leg_left/leg/ankle_left",
        mesh_prim_paths=["/World/ground"],
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.08, size=(0.08, 0.0)),
        max_distance=1.0,
        global_world_only=True,
    )
    right_foot_height = left_foot_height.replace(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base/bearing_roll/hip_l_2/upper_leg_right/leg_2/ankle_right"
    )


@configclass
class MicroDuckRoughCurriculumCfg(CurriculumCfg):
    """Introduce action smoothing before harder standing, head and COM objectives."""

    terrain_levels = CurriculumTermCfg(func=microduck_terrain_levels)

    action_rate_weight = _schedule(
        "rewards.action_rate_l2.weight", (750, -0.2), (1000, -0.4), (1250, -0.6), (1500, -0.8), (1750, -1.0)
    )
    head_pose_bias_weight = _schedule("rewards.head_pose_bias.weight", (1100, -1.0), (1500, -2.0), (2000, -3.0))
    standing_envs = _schedule(
        "commands.base_velocity.rel_standing_envs", (1000, 0.05), (1250, 0.1), (1500, 0.15), (2000, 0.2), (2500, 0.25)
    )
    head_pose_range = _schedule(
        "commands.head_pose.ranges",
        (1000, ((-0.17, 0.17), (-0.17, 0.17), (-0.21, 0.21), (-0.047, 0.047))),
        (1500, ((-0.39, 0.39), (-0.39, 0.39), (-0.49, 0.49), (-0.11, 0.11))),
        (2000, ((-0.72, 0.72), (-0.72, 0.72), (-0.91, 0.91), (-0.2, 0.2))),
        (2500, ((-1.1, 1.1), (-1.1, 1.1), (-1.4, 1.4), (-0.31, 0.31))),
    )
    com_range = _schedule(
        "events.randomize_com.params.com_range",
        (1000, {axis: (-0.005, 0.005) for axis in "xyz"}),
        (1500, {axis: (-0.01, 0.01) for axis in "xyz"}),
        (2000, {axis: (-0.015, 0.015) for axis in "xyz"}),
    )
    head_com_range = _schedule(
        "events.randomize_head_com.params.com_range",
        (1000, {axis: (-0.005, 0.005) for axis in "xyz"}),
        (1500, {axis: (-0.01, 0.01) for axis in "xyz"}),
    )


def rough_terrain_generator():
    """Build the four-tile terrain mixture for blind walking."""
    return terrain_gen.TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=10,
        num_cols=20,
        curriculum=True,
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25),
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.25,
                step_height_range=(0.0, 0.015),
                step_width=0.15,
                platform_width=2.0,
                border_width=1.0,
            ),
            "random_grid": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.30, grid_width=0.45, grid_height_range=(0.0, 0.010), platform_width=1.5
            ),
            "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.20, slope_range=(0.03, 0.10), platform_width=2.0, vertical_scale=0.001
            ),
        },
    )
