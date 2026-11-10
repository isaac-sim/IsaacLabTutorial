# 3. Play with a puppet duck

**Challenge:** make Microduck look left, crouch, or lift one foot.

```bash
uv run --project environments/locomotion python scripts/locomotion/playground.py
```

Open **Microduck playground** in the Newton sidebar. The head/neck group starts open; expand either leg to see its joints. Sliders display degrees; the simulation receives radians. Each joint has a target and a measured angle.

The default **Supported pose** fixes the root above the floor so you can inspect motion without balancing. It is an inspection configuration, not a trained locomotion task. The feet hanging clear of the ground are intentional.

Move `head_yaw`, then a knee. Predict the direction first. The slider sends a target through the real BAM actuator model; it does not teleport the joint. The measured angle may lag or settle away from the target.

**Restore stance** changes targets back to their defaults. **Reset experiment** resets physical state and controller histories. **Pause playground** freezes control stepping while preserving viewer interaction; **Single control step** advances one 20 ms control interval. **Save experiment** writes named targets and timing to `outputs/playground/experiment.json`.

## Remove support

```bash
uv run --project environments/locomotion python scripts/locomotion/playground.py --free
```

This launches a separate free-base scene. Try a small knee change. A pose that looked reasonable while supported may fall when it has to balance. Reset and try again; this activity is not a recovery controller.

## No graphical desktop

```bash
uv run --project environments/locomotion python scripts/locomotion/playground.py --headless --steps 120 --scripted --capture
```

The same actuator/control path runs a named 0.2 rad head-yaw change halfway through. Inspect `outputs/playground/receipt.json` and `playground.png`. The capture is a simulator view, not proof of clicking a live desktop widget.

**Check yourself:** why can a target slider move before the measured joint reaches it? The policy/controller requests a target; the motor, inertia, limits, and contacts determine the physical response.
