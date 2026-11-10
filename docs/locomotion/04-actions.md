# 4. Follow one action into the robot

Return to the puppet and choose one joint. Change its target slightly and inspect the measured value. Predict what happens if you reverse the target.

The tutorial preserves the robot's explicit 14-servo order: five left-leg joints, four head/neck joints, then five right-leg joints. The asset's internal order can differ. Selecting by name with `preserve_order=True` keeps observations and actions consistent.

For the selected action term:

```text
target = default_joint_position + scale × action − encoder_bias
```

The puppet inverts this mapping to turn your desired physical angle into a policy-shaped action. In nominal conditions the encoder bias is zero and the scale is one. A zero action therefore requests the reference stance; it does not mean zero torque or zero angle.

Physics advances every 5 ms (200 Hz). The policy/control interval contains four physics steps, so actions update every 20 ms (50 Hz). A held position target is still processed by the actuator during the intervening physics steps.

Compare a supported pose with a free-base trial. Actuator response and balance are different problems: reaching one requested angle does not ensure that the centre of mass stays supported.

**Experiment:** request a small head movement, restore stance, and repeat with a larger movement. Record target and measured values. Never infer torque, speed, or ideal tracking merely from the slider's value.

**Reference code:** `JointControls.actions()` implements the inverse mapping; the maintained task's `BiasedJointPositionAction` applies it. The source robot config owns the mechanical/actuator parameters.
