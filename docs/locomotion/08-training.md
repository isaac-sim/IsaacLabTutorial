# 8. Investigate the first attempt

Return to the checkpoint and logs from lesson 2. You do not need to repeat an unchanged smoke run.

```bash
uv run --project environments/locomotion python -m tensorboard.main --logdir logs/rsl_rl/microduck_tutorial
```

Inspect reward, mean episode length, velocity errors, exploration scale, and ending reasons together. A high upright reward can coexist with poor command tracking. Short episodes and a high fall fraction are evidence to inspect the robot/controller/task, not an automatic reason to increase the training budget.

To test a larger budget, change only `--max_iterations` in the first-run command and give it a new `--run_name`. Keep the seed, environment count, reward definition, observations, and action mapping fixed. Record actual elapsed collection/update time separately from compilation and download time.

The initial nominal task disables extra sensor corruption, command/observation delay, battery variation, pushes, and training curricula while retaining the actual BAM motor model. This is a teaching baseline, not an ideal instantaneous actuator. Head/body commands remain fixed.

**Expected evidence:** a reloadable checkpoint, finite diagnostics, and real playback. The first validated preview checkpoint fell early; the tutorial does not supply that as a successful walking controller. A successful reference policy requires a longer validated run and fixed-command evaluation.

**Record:** identify one plausible failure cause and one observation that would distinguish it from another cause. Change one factor in the next trial.
