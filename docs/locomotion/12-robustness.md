# 12. Teach the duck about uncertainty

The [measured reality-gap matrix](09-evaluation.md#find-the-failing-condition) exposed a problem: the nominal policies fell with added observation lag. Predict whether training with varied lag will help. Start a bounded fresh delay-trained policy:

```bash
uv run --project environments/locomotion isaaclab train --rl_library rsl_rl --task MicroDuckTutorial-Velocity-Flat-Delay --num_envs 128 --max_iterations 50 --seed 42 --run_name delay_preview --visualizer none
```

Replay its exact saved checkpoint with `--training-profile delay`. A short run can fail before reaching the walking commands; keep that result. The profile flag checks the saved training task and complete environment configuration before reconstructing the chosen evaluation conditions. Equal tensor widths alone do not qualify a checkpoint.

```bash
uv run --project environments/locomotion python scripts/locomotion/evaluate.py --training-profile delay --checkpoint logs/rsl_rl/microduck_tutorial/YOUR_DELAY_RUN/model_49.pt --start-protocol balance-starts-v1 --seed 0 --observation-delay-ms 40 --output outputs/delay-trained-check
```

## Change the training distribution, keep the test fixed

Use the same physical-start seed, command schedule, and test condition for each policy. Training randomization and evaluation intervention are different inputs:

```mermaid
flowchart LR
    D[Sample the declared training delays] --> L[Collect experience and update PPO]
    L --> P[Freeze the learned policy]
    C[Choose a fixed test condition and physical start] --> E[Run independent evaluation]
    P --> E
    E --> R[Keep response, coverage, falls and invalid attempts]
```

The controlled configurations form a two-factor comparison:

| Training profile | Task suffix | Robot | Training delays |
| --- | --- | --- | --- |
| `nominal` | none | Nominal asset | None |
| `delay` | `-Delay` | Nominal asset | Targets 0–30 ms; IMU/dq 0–40 ms |
| `backlash` | `-Backlash` | Known series-play asset | None |
| `combined` | `-Combined` | Known series-play asset | Same delays as `delay` |

All four retain the same PPO defaults, reward weights, commands, control timing and actor information. Noise, encoder bias, supply variation, mass/friction randomization, pushes and curricula stay disabled in this comparison. The backlash factor selects the complete known asset, including its tiny additional passive-body masses; the [asset lock](evidence/asset-lock.json) records that distinction.

Target lag is an integer draw in 5 ms physics steps, redrawn every physics step. Joint-velocity lag is an integer draw in 20 ms samples, held until episode reset. IMU gyro/gravity lags have separate phases and are redrawn every 64 control samples. Startup history can still return fresher data. These are declared teaching distributions, not measured hardware timing tolerances.

Forty milliseconds lies inside the sensor-delay training range. Success there tests interpolation under a fixed condition; it does not establish extrapolation to longer delays. Try 60 ms as a separately declared test after inspecting the in-range result.

## Repeat the controlled experiment

For a meaningful comparison, use the same full transition budget and at least three independent training seeds per profile. The automated driver trains one replicate, checks actual checkpoint progress, then evaluates nominal, 40 ms lag, backlash, and backlash-plus-lag conditions under evaluation seeds 0–9:

```bash
uv run --project environments/locomotion python scripts/locomotion/run_robustness_trial.py --profile combined --seed 42 --num-envs 4096 --iterations 1000 --output outputs/combined-seed42
```

Repeat for the other profiles and training seeds 0 and 1. At 24 rollout steps this requests 98,304,000 fresh transitions per replicate. The driver records the actual completed iterations and optimizer counters, not just the requested limit. It preserves every evaluated outcome, resolved parameters, event logs, source snapshot, locked robot assets, and a compressed collection archive. An execution error leaves the study incomplete; a physical fall remains an evaluated outcome.

Use `qualify.py --training-profile PROFILE --effects SNAPSHOT.json` to qualify existing checkpoints under the same frozen hardware condition. Choose one profile per seed matrix. The saved effect snapshot is checked again when resuming; changing it creates a different experiment.

Compare per-training-seed response-pass and survival rates, their variation, and complete-trial tracking errors. Keep invalid and failed attempts in denominators. A policy that stands still can survive every trial and still fail the locomotion checks. The full driver is a runnable experiment, not a promise that the randomized policy will improve.

## Look at the learned difference

In [the native effects workshop](11-effects.md), pass `--training-profile delay` (or `backlash`/`combined`) with that policy's exact checkpoint. The same frozen policy then encounters different evaluation conditions. To compare two separately trained policies, evaluate both with the same effect file and start, then pass their two receipts to `compare.py`. Keep both objectives and independent task metrics visible.

<details>
<summary>Beyond this controlled ablation</summary>

The maintained full transfer task introduces additional uncertainty and curricula:

```bash
uv run --project environments/locomotion isaaclab train --rl_library rsl_rl --task MicroDuckTutorial-Velocity-Flat-Transfer --num_envs 128 --max_iterations 50 --seed 42 --run_name transfer_preview --visualizer none presets=backlash
```

This is a separate bounded execution preview. Its masses, friction, motor/sensor effects, pushes and curricula do not constitute the same two-factor experiment. Its checkpoint requires its own declared evaluator contract; the controlled-profile flag does not admit it. Record each parameter's units, bounds, sampling cadence, measurement or justification, and schedule before drawing a transfer claim.

Observation history is another explicit architecture experiment: it changes inputs, reset state and deployment requirements. Add it during training and verify the matching export/runtime rather than adding it only on the robot.

</details>

**Record:** which uncertainty was sampled per robot, per episode or over time, and whether the candidate improved the same held-fixed test without losing nominal behavior.
