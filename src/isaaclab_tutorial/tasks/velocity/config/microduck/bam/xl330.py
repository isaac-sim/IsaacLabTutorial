# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fitted motor parameters for the Dynamixel XL330 servo."""

from .bam_cfg import BamMotorCfg

XL330_MOTOR_CFG = BamMotorCfg(
    model="m6",
    kt=0.36601349688984386,
    resistance=2.8113923539223227,
    error_gain=0.0028773775022263564,
    max_pwm=1.0,
    max_current=1.75,
    friction_base=0.004771183165566,
    friction_viscous=0.005359668274599504,
    friction_stribeck=0.004676345799486616,
    dtheta_stribeck=2.890372094130307,
    alpha=8.683259907618984,
    load_friction_motor=0.2667860954283698,
    load_friction_external=8.515871897059342e-06,
    load_friction_motor_stribeck=1.0722918395099123e-05,
    load_friction_external_stribeck=0.08077928978935671,
    load_friction_motor_quad=0.009972471242139415,
    load_friction_external_quad=0.004902565732332559,
)
"""XL330 m6 fit from Rhoban BAM revision ``62bd8ce12154340be97e06f7f41a0ca8f116d967``."""
