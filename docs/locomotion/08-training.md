# 8. Investigate the first attempt

**Challenge:** predict what doubles when you change from 128 to 256 ducks. Then compare the prediction with two real runs.

## Open the investigator

Start with the run you trained in lesson 2:

```bash
uv run --project environments/locomotion python scripts/locomotion/training_workshop.py --run latest
```

Open the printed `outputs/training-investigator/index.html` in a browser. No server or GPU is needed to use the generated report. `latest` selects the newest nominal tutorial checkpoint's run. After selecting a run, use its exact directory to keep a later experiment from changing the selection.

If you have not trained yet, explore the supplied recorded comparison:

```bash
uv run --project environments/locomotion python scripts/locomotion/training_workshop.py --report docs/locomotion/evidence/training-comparison.json
```

This rebuilds an interactive report from actual captured data. It does not include the authors' checkpoint weights. The replay command appears only when a local checkpoint matches the recorded hash.

## Predict, then move a control

1. Choose the first attempt as the recorded run. Before changing the count, calculate `128 × 24` transitions per rollout.
2. Choose **256** parallel ducks. Which numbers double? Which stay fixed?
3. Open **Prepare this separate fresh training run** and save the candidate plan. It records your prediction and the baseline contract/hashes. Supply the downloaded file's path to the printed launcher command. It verifies the contract again before executing a fresh run; the browser does not start training.
4. Return to **128**. Predict what ten optimizer epochs would change. More reuse changes optimizer work; it does not collect extra environment transitions.

![The real budget calculator shows that 256 ducks collect 6144 transitions per rollout and 307200 in fifty iterations while keeping 1000 optimizer steps](media/training-budget.jpg)

The calculator changes one factor: parallel environments. Its rollout horizon, iteration limit, epochs, minibatches, and seed remain visible. A saved plan is planned evidence; only the next run's actual logs/checkpoint prove that training occurred.

The one-factor launcher is available only when the baseline matches the current nominal task and learner configuration. A custom reward, reordered observation vector, changed control timing, or different actor cannot silently revert to defaults. Those runs remain inspectable; use their matching training configuration for a candidate. Changes to these configuration values after saving a plan also stop the launcher. To inspect a saved plan without training, run `scripts/locomotion/train_candidate.py --plan` with its path and `--dry-run` through the locomotion environment.

## Run the single-factor candidate

Reuse the first 128-environment attempt. Run this new candidate:

```bash
uv run --project environments/locomotion isaaclab train --rl_library rsl_rl --task MicroDuckTutorial-Velocity-Flat --num_envs 256 --max_iterations 50 --seed 42 --run_name investigator_n256 --visualizer none
```

Keep its printed run directory. Build a comparison with `--run` repeated for your two exact directories. The tool reads resolved `params/env.yaml`, `params/agent.yaml`, TensorBoard events, and the latest numeric checkpoint. It compares hashes of the task/objective/controller settings and reports differences. This investigator supports fresh single-GPU PPO logs; it rejects missing iteration history and inconsistent global throughput instead of guessing the amount of experience.

Switch **Compare by** between **Completed iterations**, **Fresh transitions**, and **Recorded training time**. At equal iteration count, the 256-environment run has seen twice as much experience. At equal transition count, it has made fewer PPO updates. Neither axis makes those two experiments an equal-data, equal-update comparison.

Select episode length, reward, planar velocity error, and exploration scale in turn. Move **Inspect the timeline** to read actual samples. Curves are unsmoothed; a numerical gap stays visible. Invalid evaluation receipts retain their failure reason without inventing duration or tracking scores. Training velocity errors are the task logger's metrics, not the fixed-protocol evaluation RMSE. Termination diagnostics summarize stored per-environment activations; they are not a held-out fall fraction.

![Recorded training curves for 128 and 256 ducks show rising episode return and episode length, with different experience budgets and still-substantial velocity error](media/training-curves.jpg)

## Check the behavior separately

Use [the fixed evaluator](09-evaluation.md) on each exact checkpoint. To show its results inside the report, add `--evaluation` with each receipt's path. A receipt must match a selected checkpoint's SHA-256; choosing a different policy's evaluation is rejected. The report shows its seed, conditions, available duration, coverage, and failure alongside tracking error.

In the captured example, both runs used seed 42, horizon 24, five epochs, four minibatches, and 50 iterations:

| Run | Fresh transitions | Checkpoint optimizer steps | Recorded collection + update time | Fixed-protocol result |
| --- | ---: | ---: | ---: | --- |
| 128 ducks | 153,600 | 1,000 | 10.36 s | Fell at 5.14 s; 25.7% coverage |
| 256 ducks | 307,200 | 1,000 | 10.71 s | Reached 20 s; 100% coverage |

These are actual RTX 5090 measurements, excluding setup and asset downloads. [Recorded curves and hashes](evidence/training-comparison.json) and [capture provenance](evidence/training-validation.json) preserve the evidence. Hardware, caching, and workload affect timing; the table is not a speed promise.

The surviving candidate still has tracking error. Its short-run stochastic training episodes and deterministic evaluation also measure different behavior. One seed and one deterministic nominal start do not establish a reliable improvement or a reference walking benchmark. Preserve the failed attempt; do not compare its short pre-fall RMSE as if it completed every command segment.

**Record:** identify one plausible failure cause and one observation that would distinguish it from another cause. For the next run, choose one change and state its experience budget before launching.

<details>
<summary>Keep investigating in TensorBoard</summary>

```bash
uv run --project environments/locomotion python -m tensorboard.main --logdir logs/rsl_rl/microduck_tutorial
```

Inspect reward, mean episode length, velocity errors, exploration scale, losses, and ending diagnostics together. A high upright reward can coexist with poor command tracking. The nominal task disables extra sensor corruption, command/observation delay, battery variation, pushes, and curricula while retaining the BAM actuator model. Head/body commands remain fixed. Physical behavior, observed inputs, and learning diagnostics help distinguish causes; an extra training budget alone is not a diagnosis.

</details>
