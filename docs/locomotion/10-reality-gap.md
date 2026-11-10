# 10. Which simulation signals exist on the robot?

Before hardware control, map every actor field and output joint to a real interface. Record sensor source, units, coordinate frame, joint sign/order, reference position, preprocessing, capture timestamp, receive timestamp, and update rate. Identify estimated quantities such as gravity direction or joint velocity.

The critic's simulation-only state is not a deployment input. Encoder readings need calibration to the same joint coordinates used in training. A matching-looking stance does not prove signs, scale, delay, or motor dynamics are correct.

On an actual Microduck, first gather read-only calibrated sensor/timing logs. Compare one hand-moved joint and the IMU frame with the simulator. Record firmware, bus, power conditions, and control rate. This hardware-specific adapter and its measured validation are a subsequent checkpoint; the tutorial does not invent a deployment command for an unspecified robot interface.

**Challenge:** give one mismatch that can leave the duck looking correct in a photo while causing a deployed policy to fail.
