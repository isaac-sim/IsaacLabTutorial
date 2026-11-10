# 9. Evaluate what you actually care about

Run the fixed protocol on an exact checkpoint:

```bash
uv run --project environments/locomotion python scripts/locomotion/evaluate.py --checkpoint latest --seed 0
```

The schedule is 0–4 s standing; 4–8 s forward at 0.2 m/s; 8–12 s backward at 0.1 m/s; 12–16 s lateral at 0.1 m/s; 16–20 s turning at 0.5 rad/s. Every command is supplied in the task's declared frame. Use an exact checkpoint path once selected.

## Take the controls

```bash
uv run --project environments/locomotion python scripts/locomotion/playground.py --mode policy --checkpoint latest
```

Open **Microduck playground**. This loads your saved PPO actor in inference mode. Start with **Stand**, request 0.2 m/s forward, return to Stand, then try a 0.5 rad/s yaw command. Predict whether stopping will take one control step. Sliders stay inside the training command range; the measured velocities show the actual response. The policy owns the joint targets and remains frozen throughout.

Pause and use **Single control step** to inspect one transition. In the observation inspector, select `velocity_commands` to see the command supplied to the actor, or `actions` to inspect its previous request. Freezing that display leaves the controller running. If the duck falls, its final physical pose is held until **Reset experiment**; the reset starts a new trial with cleared histories.

For a desktop-free version:

```bash
uv run --project environments/locomotion python scripts/locomotion/playground.py --mode policy --checkpoint latest --headless --scripted --steps 500 --capture --output outputs/drive
```

The 10-second script asks for stand, forward, turn, stand, then explicitly resets and stands. It stops earlier if the duck falls. Its receipt records commands, measured velocities, policy actions, requested servo angles, observation widths, and checkpoint hash. This is a driving activity with its own schedule, not the fixed 20-second benchmark above.

![Recorded PPO response follows a forward command and then a turn; lateral oscillation remains visible, followed by an explicit reset at eight seconds](media/policy-driving.jpg)

This capture uses the seed-42 reference after 1,000 iterations (98,304,000 transitions). The forward response takes several steps to settle. Lateral oscillations during walking remain visible despite a zero lateral command. A short first-attempt checkpoint can fall instead; do not expect this trace from 50 iterations. [Capture provenance](evidence/playground-validation.json) identifies the policy and scripts.

```bash
uv run --project environments/locomotion python scripts/locomotion/plot_playground.py --receipt outputs/drive/receipt.json --output outputs/drive/tracking.svg
```

The viewer's native **Reset Episode** also clears the tutorial's controller and display histories. If it interrupts a transition while the scene is rendering, that transition is logged as a reset event and excluded from the motion trace. A viewer reset is not a fall.

## Read the benchmark honestly

The evaluator stops at the **first episode ending**. It does not silently restart a fallen robot and call the next episode a continuation. Each record saves the pre-step pose/velocities, following transition reward, reward-time tracking error, and individual termination reasons; an optional image is the final pre-step state rather than the autoreset stance. A `nan_state` termination invalidates the trial and returns an error after saving its diagnostic receipt. It is not counted as an ordinary fall, even if autoreset made the subsequent state finite.

Read tracking RMSE **together with** the fall flag, available duration, and fraction of the 20-second horizon covered. Low error over a short pre-fall segment is not full-protocol success. An early-standing failure has never attempted the later walking segments.

For publication-quality claims, compare at least three independent training seeds under a declared distribution of initial states and conditions, reporting all attempts and variation across training seeds. The nominal teaching reset is deterministic: changing its seed alone does not produce ten distinct physical starting states. The one-policy preview here is exploratory. Add per-segment tracking, standing drift, and action-rate summaries before claiming a reference locomotion benchmark.

The receipt includes commanded/measured velocities, root/joint poses, per-term reward rates, original weights, seed, checkpoint hash, and timing. Keep it with comparison plots so a visual result remains auditable.

**Challenge:** find a policy with a good-looking scalar score but insufficient episode coverage. Explain why reward and task success disagree.
