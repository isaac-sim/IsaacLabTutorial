# 5. Become an observation detective

**Predict:** which signals change when you pitch the head, and which change only when the whole body tilts?

```bash
uv run --project environments/locomotion python scripts/locomotion/playground.py --mode observations
```

Open **Observation detective (read only)**. Move a joint and inspect its named slices. Pause the playground to read one state. The values are assembled environment observations; actor normalization is a separate part of the learned model.

The 61-value actor vector contains angular velocity and projected gravity (6), joint-position offsets (14), joint velocities (14), previous actions (14), desired velocity (3), head-pose commands (4), and body-pose commands (6). Head/body command slots are retained but held at zero initially.

Projected gravity expresses the world down direction in the robot/sensor frame. Tilting the robot changes its components even when its world position stays fixed. Joint motion can change position/velocity signals without rotating the base; do not expect a head movement alone to change a base-mounted IMU's gravity direction.

The critic receives 76 values, including simulation-only linear velocity and foot information. It learns a value estimate during training. The deployed actor must not silently receive those privileged values.

Command velocities use the task's declared heading/world conventions. Forward/lateral linear tracking is measured in the heading frame; yaw tracking uses world angular velocity about Z. Units are m/s, rad/s, and radians unless explicitly converted for display.

Observation inspection never removes a signal from the policy. Removing a field changes the model contract and requires a new architecture/configuration and training run.

**Record:** one signal that changed, one that stayed fixed, and why. Compare supported and free-base scenes if you need a body tilt.
