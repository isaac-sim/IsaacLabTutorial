"""SO-101 command conventions shared by the environment and reset generator."""

from __future__ import annotations

import math

# The workshop places the horizontal vial at a fixed tabletop heading.  Keep a
# small, explicit setup tolerance for this tutorial, and share it between the
# generated phase-zero resets and full-horizon evaluation.  This avoids an
# accidental train/evaluation distribution shift while remaining more varied
# than the demonstrated real setup.
TABLETOP_VIAL_HEADING_RANGE = (-0.15, 0.15)
# A narrow deadband prevents numerical chatter around zero while retaining
# Isaac Lab's standard signed binary-gripper convention. The deployed policy
# is deterministic, so a larger threshold only creates an unnecessary flat
# region in the learning problem.
GRIPPER_ACTION_THRESHOLD = 0.05


def workshop_gripper_position(raw_position: float) -> float:
    """Convert the real SO-101 gripper's 0--100 command to USD radians.

    The workshop interface maps the real robot's normalized gripper command
    linearly onto the authored ``[-10, 100]`` degree joint range.  This
    conversion defines ordinary position targets only; actuator dynamics are
    always resolved directly from the Sys-ID USD.
    """
    if not 0.0 <= raw_position <= 100.0:
        raise ValueError("raw_position must be in the real robot's [0, 100] range.")
    return math.radians(-10.0 + 1.1 * raw_position)


# Representative successful workshop demonstrations use a moderately open
# aperture during approach, nearly close the real gripper around the vial, and
# open farther only after insertion. Keeping these distinct avoids sweeping a
# fully open jaw across the vial during the grasp.
PREGRASP_GRIPPER_POSITION = workshop_gripper_position(22.4)
# Round the observed 0.97 command to 1.0: the USD soft-limit safety margin is
# -8.9 degrees, while 0.97 converts to -8.933 degrees. The 0.03-degree change
# is below the real command resolution and keeps the target inside that margin.
GRASP_GRIPPER_POSITION = workshop_gripper_position(1.0)
RELEASE_GRIPPER_POSITION = workshop_gripper_position(42.7)

# Canonical operational start from ``SO101Control.initial_pose`` in the
# sim-to-real workshop. The real controller moves here on connection. Body
# joints use that interface's calibrated raw-to-USD mapping; the task keeps
# the jaw at its ordinary open approach command instead of the recorded
# zero-width startup value.
WORKSHOP_INITIAL_JOINT_POSITION = (
    -0.1221070742,
    -0.9066845838,
    0.1900876486,
    1.4797928525,
    -0.8044013083,
    PREGRASP_GRIPPER_POSITION,
)

# Ordinary joint-position command immediately before grasp closure in a
# successful workshop vial-to-rack demonstration. It only seeds the reset
# generator's multi-start IK search onto a branch demonstrated by the real
# robot; every solved command still has to pass the randomized target pose,
# USD limits, contact, and connected dynamic-rollout checks.
WORKSHOP_PREGRASP_JOINT_POSITION = (
    0.14852054,
    0.62447881,
    -0.42907611,
    1.13695024,
    -1.65368783,
    0.24228835,
)

# Sim-radian mapping of the measured follower state at the same real episode
# frame. This is a geometry calibration reference only: reset commands still
# use the leader action above and all dynamics come from the USD drives.
WORKSHOP_PREGRASP_MEASURED_JOINT_POSITION = (
    0.14830200,
    0.64005189,
    -0.42359659,
    1.13539731,
    -1.65263290,
    0.24830763,
)

# Sparse ordinary position waypoints from the same successful real episode.
# The recorded 30 Hz trajectory is deliberately downsampled to keep the
# tutorial readable; the generator interpolates these commands slowly through
# the untouched Sys-ID drives and accepts only the object states Newton
# actually reaches. Grasping segments use the safe soft-limit close target
# above instead of the recording's 0.03-degree lower command.
WORKSHOP_TASK_WAYPOINTS = {
    3: (
        (0.09114071, 0.44803178, -0.43491304, 1.15299133, -1.64141933, GRASP_GRIPPER_POSITION),
        (-0.03038024, 0.04142760, -0.33287840, 1.23753429, -1.64005544, GRASP_GRIPPER_POSITION),
        (-0.21941283, -0.20560363, -0.17836881, 1.19817807, -1.63869155, GRASP_GRIPPER_POSITION),
    ),
    # Continue to the next demonstrated command before transport. With the
    # detailed rack collider there is no proxy rail to help rotate the vial:
    # the real trajectory must make it upright through the gripper alone.
    # Repeating the ordinary target lets the original finite-stiffness drives
    # settle before the state is tested as an independent reset.
    4: (
        (-0.38650413, -0.74569668, 0.44112619, 0.59325852, -1.64960260, GRASP_GRIPPER_POSITION),
        (-0.38650413, -0.74569668, 0.44112619, 0.59325852, -1.64960260, GRASP_GRIPPER_POSITION),
        (-0.38650413, -0.74569668, 0.44112619, 0.59325852, -1.64960260, GRASP_GRIPPER_POSITION),
    ),
    5: (
        (-0.42194772, -0.85923903, 1.08686064, -0.10640755, -1.65096635, GRASP_GRIPPER_POSITION),
        (-0.41350878, -0.86077335, 1.17140360, -0.17637415, -1.65096635, GRASP_GRIPPER_POSITION),
        (-0.41013318, -0.86077335, 1.21804800, -0.19532344, -1.65096635, GRASP_GRIPPER_POSITION),
    ),
    6: (
        (-0.42025994, -0.86077335, 1.27635356, -0.27257823, -1.65096635, GRASP_GRIPPER_POSITION),
        (-0.42532332, -0.86077335, 1.27781122, -0.27257823, -1.65096635, GRASP_GRIPPER_POSITION),
        (-0.42025994, -0.86077335, 1.27781122, -0.27403589, -1.64823868, GRASP_GRIPPER_POSITION),
    ),
    7: (
        (-0.42701111, -0.86230773, 1.27781122, -0.27403589, -1.64960258, 0.16946174),
        (-0.41519656, -0.95743777, 1.29238758, -0.25945950, -1.63869155, 0.32906288),
        (-0.41013318, -1.10013280, 1.30404862, -0.11515337, -1.63869155, 0.64051750),
    ),
}
