# 7. Explain the training run you already launched

**Two-minute challenge:** open the [training investigator](08-training.md#open-the-investigator). Predict the effect of doubling the number of parallel ducks, then try its budget control. A calculation you can check makes the PPO recipe easier to read.

Find the first run's environment count, rollout horizon, and iteration number. The teaching preview uses 128 ducks, 24 control steps per rollout, and 50 iterations. Each control step lasts 20 ms; the physics runs four 5 ms substeps inside it.

```mermaid
flowchart LR
    P[Current stochastic actor] --> R[Collect N × H transitions]
    R --> V[Critic: estimate returns and advantages]
    V --> U[Optimize minibatches for E epochs]
    U --> P
    U --> C[Save a checkpoint]
    C --> F[Freeze actor for driving and evaluation]
```

The loop returns to fresh simulation experience after the limited update phase. In the frozen driving/evaluation branch, there are no optimizer steps.

PPO repeats four operations: collect a recent rollout with the current stochastic actor; estimate returns and advantages with the critic; update actor and critic for a limited number of minibatch epochs; collect fresh experience.

An advantage estimates whether an action worked better than expected from its state. Positive advantage encourages that action; negative advantage discourages it. The probability ratio compares its likelihood under the candidate and rollout policies. PPO clips the surrogate objective to limit the incentive for large changes. Clipping is not a guarantee that every action probability stays inside a hard interval.

**Predict before opening:** for a positive advantage, does making an action twice as likely keep improving the clipped objective? What changes for a negative advantage?

<details>
<summary>Inspect the actual objective shape</summary>

![Analytic PPO objective: positive advantages flatten above ratio 1.2, while negative advantages flatten below 0.8; the unclipped objective remains a straight line](media/ppo-objective.jpg)

The plot maximizes `min(r × A, clip(r, 1 − ε, 1 + ε) × A)`, using `ε = 0.2` and illustrative advantages `A = +1` and `A = −1`. Here `r = exp(new_log_probability − rollout_log_probability)`. RSL-RL minimizes the negative of this expression. These are analytic curves, not training measurements. The clipped and unclipped lines overlap on the side where the objective still discourages a harmful change. [Figure inputs and source](evidence/ppo-objective.json).

Rebuild with `uv run --project environments/locomotion python scripts/locomotion/plot_ppo_objective.py`. Keep the chosen epsilon with your figure when exploring another value.

</details>

<details>
<summary>Work through one advantage building block</summary>

For an illustrative transition with reward `r = 0.3`, value `V(s) = 1.0`, next value `V(s') = 0.9`, and discount `γ = 0.99`:

| Transition | Next-value bootstrap | One-step TD residual |
| --- | --- | --- |
| Ordinary continuing step | `0.99 × 0.9` | `0.3 + 0.891 − 1.0 = 0.191` |
| True terminal fall | `0` | `0.3 − 1.0 = −0.7` |
| Time limit in this pinned runner | Add `0.99 × V(s) = 0.99` to reward | `0.3 + 0.99 − 1.0 = 0.29` |

The residual compares a one-step target with the critic's previous estimate. GAE combines residuals across multiple steps using `γ` and `λ`; this table is not the full GAE calculation. For this continuing task, RSL-RL 5.5.1 handles `time_outs` by adding `γ × V(s)` from the stored pre-step value to the reward, an approximation to bootstrapping the final pre-reset state. True terminations receive no such addition. Do not describe that implementation as evaluating the final observation or bootstrap from an already reset state. The evaluator separately preserves the physical terminal state for diagnosis.

</details>

The value loss trains the critic; entropy encourages exploration; gradient clipping limits an optimizer step. The configured horizon and number of parallel environments determine the rollout size. Epochs/minibatches reuse that recent rollout without creating additional environment transitions.

<details>
<summary>Follow a fall and a time limit through reset</summary>

```mermaid
flowchart TD
    Step[Physics step and reward] --> End{Episode ending?}
    End -->|No| Continue[Record transition and continue]
    End -->|Fall or time limit| Final[Preserve pre-reset state and ending flags]
    Final --> Reset[Reset physical state and observation/action histories]
    Reset --> New[First observation of the new episode]
    Final --> Flags[Send termination and truncation flags to runner]
    Flags -->|Fall| Terminal[No continuing-value addition]
    Flags -->|Time limit| Timeout[Add discounted stored pre-step value to that transition reward]
```

The evaluator holds the first failed pose rather than letting autoreset conceal it. During training, the environment autoresets before returning observations to the runner. Ending flags preserve the distinction; the time-limit addition uses the runner's stored pre-step value, not the new reset state's value. See the numerical example above for this pinned runner's approximation.

</details>

The maintained PPO recipe uses 24 control steps per environment, discount 0.99, GAE lambda 0.95, clip parameter 0.2, five learning epochs, and four minibatches. A rollout of 128 × 24 contains 3,072 transitions. Fifty iterations collect 153,600 transitions, regardless of how many optimizer minibatches reuse them.

Each epoch splits the rollout into four minibatches of 768 samples. Five epochs give 20 optimizer steps per iteration, or 1,000 over the preview run. The optimizer jointly updates actor and critic; those are not two separate counts of fresh experience. The investigator reads optimizer counters from the actual saved checkpoint alongside the arithmetic prediction.

**Experiment:** change only the number of environments in a new short run. Predict the new experience budget and memory use direction. A changed transition budget is a changed experiment; do not call it an equal-data comparison.

**Check yourself:** why do we collect fresh data after the PPO updates? PPO learns from recent experience sampled by its rollout policy; indefinitely reusing old data changes that relationship.
