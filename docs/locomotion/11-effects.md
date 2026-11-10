# 11. Give the policy stale observations

Predict the effect of a 40 ms observation delay, then compare matching evaluations:

```bash
uv run --project environments/locomotion python scripts/locomotion/evaluate.py --checkpoint latest --seed 0 --output outputs/baseline
uv run --project environments/locomotion python scripts/locomotion/evaluate.py --checkpoint latest --seed 0 --observation-delay-ms 40 --output outputs/observation-delay
```

Use the same exact checkpoint, seed, and commands. The script changes IMU and joint-velocity observation lag; joint-position and previous-action channels are unchanged. It is a specific sensor-delay experiment, not a claim that every input is delayed.

Command delay is separate. Try `--command-delay-ms 20`: it delays actuator targets by four 5 ms physics steps. A 40 ms observation delay spans two 20 ms samples. Converting both from the same step count would be wrong. Histories have startup/reset behavior; inspect and verify it before interpreting the first samples as steady-state latency.

Backlash is state-dependent mechanical play, particularly visible after reversals. It differs from fixed encoder bias, random noise, an action deadband, and pure delay. The maintained transfer task offers a backlash asset through `presets=backlash`; that changes the asset/observation mapping and requires scene reconstruction. A future sandbox control must not pretend the asset can be replaced by changing a scalar on an active scene.

The reference-ghost activity is a subsequent checkpoint: record or simulate the baseline, then compare a solid changed-condition duck under synchronized initial conditions and commands. It is a comparison of closed-loop trajectories, whose observations and actions may diverge. Displaying a ghost must not affect physics.

**Record:** which signals were delayed, the delay in milliseconds and samples, and whether the policy survived long enough for the change to be interpretable.
