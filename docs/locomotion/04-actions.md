# 4. Follow one action into the robot

**Try:** open a supported head-yaw experiment. No checkpoint is needed.

```bash
uv run --project environments/locomotion python scripts/locomotion/actuator_workshop.py --output outputs/my-reversal
```

In **Supported actuator workshop**, choose a small amplitude and press **Reset and run reversal**. The duck holds, sweeps, reverses, then returns its target home over three simulated seconds. Compare requested target, actual delayed target and measured servo angle in the native viewer's live plots. Pause at a control boundary or take a single control step. Try another amplitude; it applies to the next run, which is saved separately.

**Predict:** when the target returns home, must the joint already be home? The motor has dynamics and friction. The slider requests motion; it does not prescribe the measured trajectory. [Add delay and mechanical play](11-effects.md#watch-a-bounded-reversal) once you can identify these signals. Add `--headless` to record one complete trial without a viewer.

The tutorial preserves the robot's explicit 14-servo order: five left-leg joints, four head/neck joints, then five right-leg joints. The asset's internal order can differ. Selecting by name with `preserve_order=True` keeps observations and actions consistent.

For the selected action term:

```text
target = default_joint_position + scale × action − encoder_bias
```

The puppet inverts this mapping to turn your desired physical angle into a policy-shaped action. In nominal conditions the encoder bias is zero and the scale is one. A zero action therefore requests the reference stance; it does not mean zero torque or zero angle.

Physics advances every 5 ms (200 Hz). The policy/control interval contains four physics steps, so actions update every 20 ms (50 Hz). A held position target is still processed by the actuator during the intervening physics steps.

```mermaid
flowchart LR
    A[One raw policy action] --> Scale[Multiply by configured scale]
    Scale --> Home[Add home angle, subtract encoder bias]
    Home --> Target[Requested position target, rad]
    Target --> Servo[BAM servo model at each physics step]
    Servo --> Physics[Joint dynamics and contacts]
    Physics --> Encoder[Measured position and velocity]
    Encoder --> Next[Next actor observation]
```

The nominal case has no command lag. With lag enabled, the actuator's delay buffer selects an older target before computing its response. Mechanical limits, effort limits and servo dynamics belong to the actual configured model; the action formula alone does not guarantee a reachable target.

| Simulation time | 0 ms | 5 ms | 10 ms | 15 ms | 20 ms |
| --- | --- | --- | --- | --- | --- |
| Actor input/action | Read state; produce action A | — | — | — | Read new state; produce B |
| Requested target | A | Hold A | Hold A | Hold A | B |
| Actuator and physics | Advance 0–5 | Advance 5–10 | Advance 10–15 | Advance 15–20 | Advance 20–25 |

**Predict:** can an observation that arrives 40 ms late be repaired by running physics four times per control interval? More physics steps do not make an old measurement newer; inspect this distinction in [the reality-gap sandbox](11-effects.md).

Compare a supported pose with a free-base trial. Actuator response and balance are different problems: reaching one requested angle does not ensure that the centre of mass stays supported.

**Experiment:** request a small head movement, restore stance, and repeat with a larger movement. Record target and measured values. Never infer torque, speed, or ideal tracking merely from the slider's value.

<details>
<summary>Change action scale while keeping raw actions fixed</summary>

For `head_yaw`, action slot 7 has home angle 0 rad. A raw action of 0.10 requests 0.10 rad at scale 1, or 0.05 rad at scale 0.5. For `right_knee`, slot 12 has home 0.0049 rad: that same raw action requests 0.1049 or 0.0549 rad before bias compensation. The joint order comes from `DUCK_JOINTS` in `locomotion/deployment.py`.

Predict which trace changes when only scale changes, then run the same bounded raw-action program twice:

```bash
uv run --project environments/locomotion python scripts/locomotion/actuator_workshop.py --amplitude-deg 4 --action-scale 1 --output outputs/scale-one
uv run --project environments/locomotion python scripts/locomotion/actuator_workshop.py --amplitude-deg 4 --action-scale 0.5 --output outputs/scale-half
```

Run them one at a time and press **Reset and run reversal** in each. Add `--headless` to finish automatically. The amplitude control expresses raw action in degrees at scale 1; the panel separately shows the resulting requested-target amplitude. Scale 0.5 halves that target offset, while the measured response still comes from the real servo model. Each output's `trial-000/receipt.json` records the scale and native target samples.

Changing a puppet target alone would not isolate this question: the puppet inverts the mapping to reach your chosen physical angle. This experiment holds raw actions fixed instead. Scale is set before building the scene; it does not change while a reversal runs. The trained tutorial policy retains scale 1.

**Record:** which quantity halved, and why the measured joint trajectory need not halve exactly.

</details>

**Reference code:** `JointControls.actions()` implements the inverse mapping; the maintained task's `BiasedJointPositionAction` applies it. The source robot config owns the mechanical/actuator parameters.
