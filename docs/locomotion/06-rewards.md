# 6. Can a better score hide worse walking?

First evaluate a trained checkpoint so there is a real trajectory to inspect:

```bash
uv run --project environments/locomotion python scripts/locomotion/evaluate.py --checkpoint latest
uv run --project environments/locomotion python scripts/locomotion/reward_workshop.py --trajectory outputs/evaluation/receipt.json
```

The reward workshop replays **recorded motion**. Move reward-weight sliders and watch the original/candidate scores. Change velocity tolerance to make the tracking curve narrower or broader. These controls rescore the same states; they do not update a policy or generate alternate movement.

The manager evaluates each active term, multiplies it by its configured weight, sums the results, then multiplies by the 0.02 s control interval. Recorded per-term rates are converted back to raw terms before applying candidate weights. Terms disabled during collection cannot be reconstructed and are not offered as new controls.

The linear tracking reward is `exp(-squared_heading_velocity_error / std²)`. Width changes recompute this term from the squared error recorded at the same post-physics instant as the reward, before autoreset. The evaluator's separate pre-step metric velocities are not substituted for that reward-time state. Merely multiplying the previous score would be incorrect.

<details>
<summary>Predict what changing the tolerance does</summary>

![Analytic tracking reward at the actual 0.15 m/s tolerance and an illustrative 0.30 m/s alternative; a second panel includes weight 4 and the 0.02-second control interval](media/reward-shape.jpg)

At the same nonzero velocity error, a larger width gives a larger tracking reward. Both curves equal one at zero error; increasing tolerance does not create better tracking. The right panel shows this term's contribution after its actual weight and control interval are applied. These are equation curves using [recorded nominal parameters](evidence/reward-shape.json), not learned-policy measurements. Rebuild with `uv run --project environments/locomotion python scripts/locomotion/plot_reward_shape.py`.

Try the workshop's width slider, then explain why making the score easier to earn could leave task performance unchanged.

</details>

## Predict what stronger smoothness will teach

Make the `action_rate_l2` weight ten times more negative: −0.1 → −1.0. Before retraining, predict whether the duck will walk more smoothly, stop walking, or fall. The penalty measures changes in consecutive raw actions; it does not directly measure motor torque or energy.

We actually trained this candidate from scratch with the same seed and budget as the reference: 4096 environments × 24 steps × 1000 iterations, or 98,304,000 transitions per policy. The saved task/learner recipes differ only in that reward weight. Both were evaluated on the same `balance-starts-v1` initial state and 20-second commands.

![Two actual learned motions scored under both objectives; the stronger-penalty policy survives but scarcely translates, while the baseline tracks commands](media/reward-comparison.jpg)

Both policies stayed upright for 20 seconds. The candidate's forward/backward/lateral response gains were approximately zero, so it failed the locomotion criteria. Its requested-target rate RMS fell from 8.17 to 0.72 rad/s, but the baseline earned more under **both** objectives: 251.41 versus 197.68 under the original, and 244.69 versus 197.62 under the candidate objective. This is a poor learned result, even under its own objective; a changed reward does not guarantee successful optimization. It is one training seed and one physical start, not an aggregate robustness result. [Measurements and provenance](evidence/reward-validation.json).

![The blue baseline ghost has travelled while the solid stronger-penalty duck remains near its start; both paths retain their world coordinates](media/reward-policy-ghost.jpg)

With those local recordings installed, explore the actual motions:

```bash
uv run --project environments/locomotion python scripts/locomotion/compare.py --reference outputs/ppo-starts-matrix/trials/train-seed-42/eval-seed-0/receipt.json --candidate outputs/reward-action-rate-candidate/receipt.json
```

The two actors generated their own observations and actions during independent simulations. The ghost only replays them. To rescore your own recording with this candidate objective, add `--candidate docs/locomotion/experiments/reward-action-rate.json` to the reward-workshop command. Inspect the [per-term contributions](media/reward-terms.jpg): which component was reduced, and which tracking reward was lost?

<details>
<summary>Reproduce the longer measured training run</summary>

The workshop's saved command starts with the bounded 128-environment, 50-iteration teaching preset. The measured reference comparison above used this larger equal-budget candidate:

```bash
uv run --project environments/locomotion --locked isaaclab train --rl_library rsl_rl --task MicroDuckTutorial-Velocity-Flat --num_envs 4096 --max_iterations 1000 --seed 42 --run_name reward_action_rate_x10 --visualizer none env.rewards.action_rate_l2.weight=-1.0
```

Evaluate its exact saved checkpoint with `--start-protocol balance-starts-v1 --seed 0`, then compare with the same physical start of your baseline. Keep every result; the experiment does not guarantee a particular learned outcome.

</details>

## Find a loophole

Try increasing upright/standing-related rewards while reducing velocity tracking. Can a nearly stationary trajectory earn a more attractive score? Compare task metrics before deciding that a larger return is useful.

Select **Save candidate and training command**. The workshop writes `candidate.json`, `scores.json`, a score plot, and `train-candidate.txt`. Inspect the command, then run it in a separate terminal to train a new candidate with an explicit configuration. New training is what can change behavior.

For headless rescoring:

```bash
uv run --project environments/locomotion python scripts/locomotion/reward_workshop.py --trajectory outputs/evaluation/receipt.json --headless
```

Add `--candidate outputs/reward-workshop/candidate.json` to reload saved weights. Add `--capture` with `--headless` to render the recorded pose stream.

Once candidate training writes a checkpoint, compare the two actual policies with a ghost:

```bash
uv run --project environments/locomotion python scripts/locomotion/reality_gap.py --checkpoint logs/rsl_rl/microduck_tutorial/BASELINE_RUN/model_49.pt --candidate-checkpoint logs/rsl_rl/microduck_tutorial/CANDIDATE_RUN/model_49.pt --seed 0
```

Replace both paths with saved checkpoints. Reward-only rescoring keeps one motion stream; this command runs two learned policies. Compare both recorded trajectories under both objectives using the workshop and retain independent tracking/fall metrics. Do not compare only each policy's preferred reward. See [the ghost controls](11-effects.md) for synchronized playback and early-ending markers.

The evaluator uses the common nominal reward configuration when comparing policies. A candidate's training objective is recovered explicitly from its saved parameters, then applied to both recordings during rescoring. The published [numerical comparison](evidence/reward-comparison.json) retains both weight sets. Rebuild its figures without the larger local recordings using `uv run --project environments/locomotion python scripts/locomotion/plot_reward_comparison.py`.

**Record:** “I increased the score by …; this does/does not demonstrate improved walking because ….”
