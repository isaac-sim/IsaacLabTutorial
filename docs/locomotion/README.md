# Play, train, and understand Microduck

Teach a small biped to follow velocity commands on flat ground with PPO. Start by trying things, then explain the result. A duck that falls gives you something concrete to investigate.

This development path uses an **opt-in environment** pinned to a public Microduck/BAM source revision and its matching Newton/MuJoCo packages. The repository's root environment continues to serve the SO-101 tutorial. The Microduck task is not assumed to exist in the released Isaac Lab wheel.

Run all commands below from the **repository root**. Python 3.12, uv, and a compatible NVIDIA GPU are required for simulation/training. Native sliders need a graphical desktop; each activity also has a headless route.

## Your first session

1. [Set up and try training immediately](02-first-training.md). No PPO theory prerequisite.
2. [Play with the puppet duck](03-puppet.md): move joint sliders and compare pose with balance.
3. [Follow a command through the actuator](04-actions.md).
4. [Investigate observations](05-observations.md) and [experiment with rewards](06-rewards.md).
5. [Understand PPO](07-ppo.md), [play with the batch calculator and compare real training curves](08-training.md), then [explore the measured trial matrix and drive the frozen policy](09-evaluation.md).
6. [Map the reality gap](10-reality-gap.md), [investigate delays/backlash](11-effects.md), [train for robustness](12-robustness.md), and [export for deployment](13-deployment.md).
7. Finish with the Duck adaptations of [SuperMimic](14-super-mimic.md) and [FlashSAC](15-flash-sac.md), using the separate pinned workflows environment.

The runnable path includes supported/free-base joint control, observation inspection, a bounded fresh PPO run, failure-aware evaluation, reward rescoring/retraining, independent reference-ghost comparisons, and executed JIT/ONNX inference checks. A short run is not a validated walking controller. The exploratory lessons preserve their measured successes and failures; the qualified SuperMimic recipe explicitly uses teacher data and initialization. Calibrated hardware transfer remains a measured checkpoint.

## How to use the lessons

**Try → predict → change one thing → observe → explain → retry.** The main path is hands-on; theory panels explain what you have already seen. Record one sentence after each experiment. Keep the seed and reset protocol fixed when comparing changes.

A **Run** block is an executable command. A **Reference** block explains code; it is not an instruction to paste it. Expected behavior describes what to look for, not a guaranteed training outcome. Checkpoint and hardware paths are inputs you choose explicitly.

Contributors can inspect [the production evidence and visual inventory](production.md) for reproducible figures, activity coverage and remaining publication gates.
