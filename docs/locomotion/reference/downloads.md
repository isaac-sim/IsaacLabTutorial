# Tutorial movies and recorded inputs

These files accompany the measured examples. They are separate from the fresh short policy you train in lesson 2. [The delivery manifest](../evidence/media-delivery.json) records every SHA256, size and purpose; the release also includes `manifest.json`.

[Public download verification](../evidence/media-publication.json) checked every file without a login against its exact size and SHA256.

| File | Contents |
| --- | --- |
| [flat-walking-overview.mp4](https://github.com/AntoineRichard/IsaacLabTutorial/releases/download/microduck-tutorial-media-3463b86/flat-walking-overview.mp4) | 20-second flat-ground command sequence |
| [training-progress.mp4](https://github.com/AntoineRichard/IsaacLabTutorial/releases/download/microduck-tutorial-media-3463b86/training-progress.mp4) | Three checkpoints with their actual training progress |
| [first-attempt.mp4](https://github.com/AntoineRichard/IsaacLabTutorial/releases/download/microduck-tutorial-media-3463b86/first-attempt.mp4) | Short fresh PPO attempt and its recorded fall |
| [stress-comparison.mp4](https://github.com/AntoineRichard/IsaacLabTutorial/releases/download/microduck-tutorial-media-3463b86/stress-comparison.mp4) | Nominal-trained and delay-trained policies under the same sensor lag |
| [supermimic-normalization.mp4](https://github.com/AntoineRichard/IsaacLabTutorial/releases/download/microduck-tutorial-media-3463b86/supermimic-normalization.mp4) | Adaptive and fixed input-normalization comparison |
| [checkpoint-progress-recordings.tar.gz](https://github.com/AntoineRichard/IsaacLabTutorial/releases/download/microduck-tutorial-media-3463b86/checkpoint-progress-recordings.tar.gz) | Recorded inputs for the overview, progress and first-attempt movies |
| [stress-recordings.tar.gz](https://github.com/AntoineRichard/IsaacLabTutorial/releases/download/microduck-tutorial-media-3463b86/stress-recordings.tar.gz) | Matched stress-comparison recordings |
| [supermimic-recordings.tar.gz](https://github.com/AntoineRichard/IsaacLabTutorial/releases/download/microduck-tutorial-media-3463b86/supermimic-recordings.tar.gz) | Original adaptive/fixed SuperMimic recordings |
| [duck-transitions.pt](https://github.com/AntoineRichard/IsaacLabTutorial/releases/download/microduck-tutorial-media-3463b86/duck-transitions.pt) | 8,192 captured Duck transitions for the workload benchmark |
| [ppo-reference-policies.tar.gz](https://github.com/AntoineRichard/IsaacLabTutorial/releases/download/microduck-tutorial-media-3463b86/ppo-reference-policies.tar.gz) | Frozen nominal/delay PPO models and run metadata |

The recordings preserve actual independent motion and failures; a movie does not run physics or update a learner. Extract the recording archives into separate directories, then use the render commands in the corresponding lessons. The robot USD is downloaded separately through the repository byte lock. The transition dataset supports the workload experiment in lesson 15; measuring it still requires your declared SAC checkpoint.

## Optional reference policies

Use this bundle after your own first attempt if you want to drive a longer-trained policy. It contains one seed-42 nominal policy and one separately trained delay policy, each with its actual checkpoint, run metadata and YAML parameters. Each collected 98,304,000 transitions and 20,000 optimizer steps. The nominal bundle is the policy used in the sensitivity study; the overview/checkpoint-progress movies identify a separate training run. These are simulated teaching policies.

From the repository root:

```bash
mkdir -p outputs/reference-policies
curl --fail --location https://github.com/AntoineRichard/IsaacLabTutorial/releases/download/microduck-tutorial-media-3463b86/ppo-reference-policies.tar.gz -o outputs/ppo-reference-policies.tar.gz
printf '454a6ed1446a46cda3b8e3acfc2922477c08801aded9aef4f3e516fe1641794a  outputs/ppo-reference-policies.tar.gz\n' | sha256sum --check -
tar -xzf outputs/ppo-reference-policies.tar.gz -C outputs/reference-policies
uv run --project environments/locomotion --locked python scripts/locomotion/playground.py --mode policy --checkpoint outputs/reference-policies/nominal/model_999.pt
```

On a headless machine add `--headless --scripted --steps 500`. Driving commands differ from the fixed qualification protocol. To recreate the delay policy's declared 40 ms sensor-lag test:

```bash
uv run --project environments/locomotion --locked python scripts/locomotion/evaluate.py --training-profile delay --checkpoint outputs/reference-policies/delay/model_999.pt --observation-delay-ms 40 --start-protocol balance-starts-v1 --seed 0 --output outputs/reference-delay-check
```

These commands run simulation. Hardware calibration, complete board timing and deployment checks remain in [lesson 13](../13-deployment.md).
