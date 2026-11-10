# 6. Can a better score hide worse walking?

First evaluate a trained checkpoint so there is a real trajectory to inspect:

```bash
uv run --project environments/locomotion python scripts/locomotion/evaluate.py --checkpoint latest
uv run --project environments/locomotion python scripts/locomotion/reward_workshop.py --trajectory outputs/evaluation/receipt.json
```

The reward workshop replays **recorded motion**. Move reward-weight sliders and watch the original/candidate scores. Change velocity tolerance to make the tracking curve narrower or broader. These controls rescore the same states; they do not update a policy or generate alternate movement.

The manager evaluates each active term, multiplies it by its configured weight, sums the results, then multiplies by the 0.02 s control interval. Recorded per-term rates are converted back to raw terms before applying candidate weights. Terms disabled during collection cannot be reconstructed and are not offered as new controls.

The linear tracking reward is `exp(-squared_heading_velocity_error / std²)`. Width changes recompute this term from the squared error recorded at the same post-physics instant as the reward, before autoreset. The evaluator's separate pre-step metric velocities are not substituted for that reward-time state. Merely multiplying the previous score would be incorrect.

## Find a loophole

Try increasing upright/standing-related rewards while reducing velocity tracking. Can a nearly stationary trajectory earn a more attractive score? Compare task metrics before deciding that a larger return is useful.

Select **Save candidate and training command**. The workshop writes `candidate.json`, `scores.json`, a score plot, and `train-candidate.txt`. Inspect the command, then run it in a separate terminal to train a new candidate with an explicit configuration. New training is what can change behavior.

For headless rescoring:

```bash
uv run --project environments/locomotion python scripts/locomotion/reward_workshop.py --trajectory outputs/evaluation/receipt.json --headless
```

Add `--candidate outputs/reward-workshop/candidate.json` to reload saved weights. Add `--capture` with `--headless` to render the recorded pose stream.

The later reference-ghost checkpoint will compare an actual retrained candidate with the baseline. Reward-only changes keep one motion stream. Compare both trajectories under both objectives and retain independent tracking/fall metrics; do not compare only each policy's preferred reward.

**Record:** “I increased the score by …; this does/does not demonstrate improved walking because ….”
