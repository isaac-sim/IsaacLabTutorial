# 12. Train under hardware uncertainty

The nominal preview is deliberately simple. The maintained transfer configuration adds sensor/encoder effects, motor and physical variation, pushes, and curricula. Fixed head/body commands are retained for this teaching route.

```bash
uv run --project environments/locomotion isaaclab train --rl_library rsl_rl --task MicroDuckTutorial-Velocity-Flat-Transfer --num_envs 128 --max_iterations 50 --seed 42 --run_name transfer_preview --visualizer none presets=backlash
```

This is a bounded execution preview; learning robust walking takes a separately validated budget. Record each randomized parameter's units, bounds/distribution, measurement or justification, sampling cadence, and schedule. Per-robot encoder bias should not be confused with independent white noise each step.

A fair experiment compares nominal, delay-only, backlash-only, and combined training under equal transition budgets and a shared evaluator. Keep the task/reward/actor information/action semantics fixed. Evaluate both nominal conditions and held-out stress conditions; report the nominal-performance tradeoff and failures as well as improvements.

Use the [measured nominal/delay matrix](09-evaluation.md#find-the-failing-condition) as the starting diagnostic: three independently trained nominal policies passed all ten physical starts without added lag, and failed all ten with 40 ms IMU/joint-velocity lag. This identifies a training intervention to investigate. It does not show that randomization has solved it. Train a candidate, then run `qualify.py` on the same starts both with and without the delay; retain failed attempts in both reports.

Start with the [named physical-start protocol](09-evaluation.md#change-the-starting-state-deliberately): run the same checkpoint with `--start-protocol balance-starts-v1` under distinct evaluation seeds. Repeat that exact sample set for each independently trained policy. Inspect the saved actual initial states; changing seed labels on `nominal-v1` does not create state diversity. Combine the same physical starts with `--observation-delay-ms` or `--command-delay-ms` to isolate a held-out delay condition. Report full command-response qualification as well as survival.

The current fixed evaluator targets the nominal-asset observation contract plus its explicit delay controls. Do not load a backlash-trained checkpoint into it and claim a matched backlash evaluation. Extend and validate the asset/preset-aware evaluator and reference-ghost sandbox before making that comparison.

Observation history is another explicit architecture experiment: it changes input dimension, reset state, and deployment requirements. Do not add history only on the robot after training.

**Challenge:** decide whether a perturbation should be fixed per robot, sampled per episode, or vary in time, and justify that choice from the physical mechanism.
