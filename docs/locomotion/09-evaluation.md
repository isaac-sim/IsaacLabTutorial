# 9. Evaluate what you actually care about

Run the fixed protocol on an exact checkpoint:

```bash
uv run --project environments/locomotion python scripts/locomotion/evaluate.py --checkpoint latest --seed 0
```

The schedule is 0–4 s standing; 4–8 s forward at 0.2 m/s; 8–12 s backward at 0.1 m/s; 12–16 s lateral at 0.1 m/s; 16–20 s turning at 0.5 rad/s. Every command is supplied in the task's declared frame. Use an exact checkpoint path once selected.

The evaluator stops at the **first episode ending**. It does not silently restart a fallen robot and call the next episode a continuation. Each record saves the pre-step pose/velocities, following transition reward, reward-time tracking error, and individual termination reasons; an optional image is the final pre-step state rather than the autoreset stance. A `nan_state` termination invalidates the trial and returns an error after saving its diagnostic receipt. It is not counted as an ordinary fall, even if autoreset made the subsequent state finite.

Read tracking RMSE **together with** the fall flag, available duration, and fraction of the 20-second horizon covered. Low error over a short pre-fall segment is not full-protocol success. An early-standing failure has never attempted the later walking segments.

For publication-quality claims, compare at least the planned three independent training seeds under the same ten evaluation seeds, reporting all attempts and variation across training seeds. The one-seed preview here is exploratory. Add per-segment tracking, standing drift, and action-rate summaries before claiming a reference locomotion benchmark.

The receipt includes commanded/measured velocities, root/joint poses, per-term reward rates, original weights, seed, checkpoint hash, and timing. Keep it with comparison plots so a visual result remains auditable.

**Challenge:** find a policy with a good-looking scalar score but insufficient episode coverage. Explain why reward and task success disagree.
