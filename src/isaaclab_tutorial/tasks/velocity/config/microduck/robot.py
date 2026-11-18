# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""The regular Microduck asset, assembled in the robot lesson."""

# BEGIN robot-joints
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_newton.sim.schemas import NewtonArticulationCfg

from isaaclab_tutorial.tasks.velocity.config.microduck.compat import install
from isaaclab_tutorial.tasks.velocity.config.microduck.compat.bam_cfg import BamActuatorCfg
from isaaclab_tutorial.tasks.velocity.config.microduck.compat.robot import MICRODUCK_CFG as REFERENCE_ROBOT_CFG

install()

MICRODUCK_JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]
# END robot-joints


# BEGIN robot-pose
STANDING_JOINT_POS = {
    ".*hip_yaw": 0.0,
    "left_hip_roll": -0.0873,
    "right_hip_roll": 0.0873,
    "left_hip_pitch": -0.4579,
    "right_hip_pitch": 0.4579,
    "left_knee": -0.0049,
    "right_knee": 0.0049,
    "left_ankle": 0.4530,
    "right_ankle": -0.4530,
    "neck_pitch": 0.3491,
    "head_pitch": 0.3491,
    "head_yaw": 0.0,
    "head_roll": 0.0,
}
# END robot-pose


# BEGIN robot-servos
SERVOS_CFG = BamActuatorCfg(
    joint_names_expr=["^(?!passive_).*"],
    motor=REFERENCE_ROBOT_CFG.actuators["servos"].motor,
    kp_fw=200.0,
    vin=7.4,
    vin_range=(6.5, 8.2),
    vin_drop_gain_range=(0.0, 0.2),
    vin_min=6.0,
    min_delay=3,
    max_delay=6,
)
# END robot-servos


# BEGIN robot-articulation
MICRODUCK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/PollenRobotics/MicroDuck/microduck_walk.usd",
        activate_contact_sensors=True,
        articulation_props=NewtonArticulationCfg(self_collision_enabled=True),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.125),
        joint_pos=STANDING_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={"servos": SERVOS_CFG},
)
# END robot-articulation
