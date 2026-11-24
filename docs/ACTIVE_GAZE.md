# Active gaze

`IsaacTutorial-Place-Vial-SO101-ActiveGaze` sees the scene through a fixed, foveated scene camera from Isaac Lab's
`isaaclab_fovi` package. Its retina samples densely where it looks and sparsely in the periphery, and the policy
moves the gaze while it moves the arm.

The task needs the `fovi` extra. Until the `isaaclab_fovi` pull request merges, the tutorial is pinned to that pull
request's branch of Isaac Lab, and one of that branch's Git LFS files is missing, so skip LFS downloads when
syncing:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --locked --extra fovi
```

In PowerShell, run `$env:GIT_LFS_SKIP_SMUDGE=1` first, then `uv sync --locked --extra fovi`.

## Variants

| Preset | Goal | Actor |
| --- | --- | --- |
| `numbered` (default) | The opening with the vial's number | CNN on the 64×64 retina, proprioception and gaze |
| `nearest` | Any opening, as in the state task | CNN on the 64×64 retina, proprioception and gaze |
| `dino_language` | The opening with the vial's number | Frozen foveated DINOv3 on the 128×128 retina, fused with an instruction |

The critic sees the privileged state, including the target opening, in every variant.

## Gaze

- The camera is an ideal square pinhole; the retina spans its field of view, so the gaze can reach anything the
  camera sees.
- The gaze is `isaaclab_fovi`'s: `GazeAction` moves it in software, turning the retina within the fixed camera's
  image, `gaze_state` observes it, and `gaze_teacher_labels` supervises it.
- Two of the eight actions move the gaze (tilt, pan). An action of 1 moves an angle by 0.1 field radii, a field
  radius being half the retina's field of view, and the gaze stays within 1 field radius of the optical axis.
  The actor observes the gaze and its velocity in the same units.
- The teacher looks at a point in the world that the task picks: the vial until the grasp, then the goal. The PPO
  loss adds the teacher's gaze actions as labels (`agent.algorithm.oracle_weight`, 1.0);
  `presets=no_gaze_supervision` turns this off.
- Episodes start the gaze at random, on the vial or the goal, or a few degrees off either
  (`env.actions.gaze.mixed_resets`).

## Train

The task trains with its own script, because action chunking wraps the environment and the reset curriculum
needs its own runner:

```bash
uv run --extra fovi python -m isaaclab_tutorial.active_gaze.train train --output runs/numbered presets=numbered,newton_warp
```

Every argument that is not a script option is an Isaac Lab configuration override, as with `isaaclab train`.
Useful presets and overrides:

- `newton_warp` renders with Newton; the default is OVRTX path-traced RGB (`--extra ovrtx`). The faster
  OVRTX Minimal modes (`minimal_full_mdl`, `minimal_diffuse_mdl`, with `_shadows` variants) draw no background.
- `sim2real` adds the tutorial's physical randomization, scene colours, a camera mount that moves each episode,
  and photometric variation of the retina. The gaze stays relative to the moved mount, and its teacher aims from
  it (`mount="tracked"`). On `dino_language`, the actor also learns to predict the mount's move.
- `chunking` makes each policy decision a plan of three control steps.
- `env.observations.vision.tokens.params.num_layers=N` runs only the first `N` DINO layers, to cut the cost of
  perception while keeping pretrained features.
- `env.scene.eye.cmf_a_deg=…` changes how strongly the retina is foveated; a large value makes it nearly uniform.

Each run writes `run.json` next to its checkpoints. It records the overrides, the retina, the gaze and the control
rates. `--checkpoint path/model_N.pt` resumes a run with its saved overrides. A new override replaces the saved
value, and `presets=…,finetune` warm-starts at a lower learning rate. Add `--logger wandb` for W&B.

## Play and inspect

```bash
uv run --extra fovi python -m isaaclab_tutorial.active_gaze.train play --checkpoint runs/numbered/model_4999.pt --output runs/numbered_play
uv run --extra fovi python -m isaaclab_tutorial.active_gaze.train inspect --num_envs 4 --output runs/inspect presets=numbered,newton_warp
```

`play` builds the run's saved configuration; a `presets=` argument would replace the saved preset list. Both write
videos and frames that show a dense camera at the scene camera's pose next to the retina. `inspect`
sweeps the gaze through every reset category without a policy, and checks that resetting one environment leaves the
others alone.

## Deploy

The `dino_language` policy runs on the real robot through a LEAPP bundle. The bundle samples the real camera's
images on the simulator's retinal grid, so describe that camera as a `fovi.sensing.projection.CameraModel` in
JSON: `model` (`pinhole` or `fisheye`), `image_size` (height, width), `intrinsics` (fx, fy, cx, cy) and OpenCV
`distortion` coefficients.

```bash
uv run --extra fovi --extra leapp python -m isaaclab_tutorial.active_gaze.deployment.export --checkpoint runs/language/model_4999.pt --camera_model camera.json --output bundles/language
uv run --extra fovi --extra leapp --with "lerobot[feetech]==0.6.1" python -m isaaclab_tutorial.active_gaze.deployment.run --bundle bundles/language --joint-map joint_map.json --port /dev/ttyACM0
```

`run` drives the arm through LeRobot, the version the tutorial's own robot scripts use. It stays read-only unless `--execute` is given with a verified joint map and a start pose, as described in
[SIM2REAL.md](SIM2REAL.md). Without `--camera_model`, the export uses the simulated camera; such a bundle can be
checked in simulation with `train.py play --deployment_bundle`, but `run` refuses to execute it.
