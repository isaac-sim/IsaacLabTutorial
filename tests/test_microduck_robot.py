"""Check the MicroDuck robot configuration assembled in lesson 02."""

from isaaclab_tutorial.tasks.velocity.config.microduck import robot


def test_joint_order_matches_policy_actions():
    leg = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"]
    head = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
    expected = [f"left_{joint}" for joint in leg] + head + [f"right_{joint}" for joint in leg]
    assert robot.MICRODUCK_JOINT_NAMES == expected


def test_robot_uses_walking_asset():
    assert robot.MICRODUCK_CFG.spawn.usd_path.endswith("/microduck_walk.usd")


def test_servo_model_and_command_delay():
    servos = robot.MICRODUCK_CFG.actuators["servos"]
    assert servos.motor.model == "m6"
    assert (servos.min_delay, servos.max_delay) == (3, 6)
