# Isaac Lab issues found while making the SO-101 vial task run on both Newton and OV PhysX

Context: Isaac Lab `develop` @ `c58715b1` (isaaclab_ov 2.0.5, isaaclab_newton, isaaclab_physx), standalone OV PhysX
runtime `ovphysx==0.5.11`, OVRTX renderer `ovrtx==0.4.1.364340`, no Isaac Sim installed (kitless). Task:
`IsaacTutorial-Place-Vial-SO101` (SO-101 `so101_new_calib_SysID` asset, workshop vial/rack/mat assets). Goal: train
on Newton (`presets=newton_mjwarp`), play on PhysX (`presets=physx`) and vice versa, changing only PhysX parameters.

Each section: what happens, how it was measured, the workaround used in this repository, and a prompt for a fix PR.
Issues are ordered by impact on sim2sim. Sections 1-2 are the earlier stand-alone reports, folded in verbatim in
spirit and shortened.


## Status of fixes in the fork

Fixes for issues 1, 4 and 5 are implemented in the Isaac Lab fork (`../mustafa_isaaclab`) on two branches with the
same patch: `sim2sim-ov-fixes-c58715b1` (commit `b00fe838`, based on the `develop` commit `c58715b1` this project
pins; this is what `pyproject.toml` now builds from via `isaaclab = { path = "../mustafa_isaaclab/tools/wheel_builder" }`)
and `sim2sim-ov-fixes` (commit `f97cc14f`, based on `develop` 18086270e, which additionally requires newton 1.6.0rc1):

- Issue 1: `OvPhysxCfg` gains `solver_type`, `friction_type`, `bounce_threshold_velocity`,
  `friction_offset_threshold`, `friction_correlation_distance`, `enable_ccd`, scene-wide min/max iteration bounds and
  `rigid_body_/articulation_position/velocity_iteration_count` defaults that are authored onto prims without their own
  value (`OvPhysxManager._author_scene_solver_attrs`, `_apply_solver_iteration_defaults`). Tests:
  `source/isaaclab_ov/test/physics/test_ovphysx_cfg_solver_settings.py`.
- Issue 4: `ContactSensor` raises a clear error naming the colliding link/joint when a filtered sensor body shares its
  name with a joint (`_find_joint_name_collisions`); the runtime-side collision itself is unchanged.
- Issue 5: `OVRTXRendererCfg.mdl_fallback` (`"unresolved"` by default) removes MDL surface outputs of materials whose
  MDL source does not resolve so their `UsdPreviewSurface` renders (`ovrtx_usd.apply_mdl_fallback`). Tests:
  `source/isaaclab_ov/test/renderers/test_ovrtx_mdl_fallback.py`.

Issues 2, 3, 6, 7, 8, 9 and 11 remain open (runtime semantics or model differences that cannot be fixed in Python).

---

## 1. OV PhysX default solver iteration counts make the jaw/vial pinch under-converged (code-level default)

**What happens.** With every joint parameter, mass, inertia, COM, collider, material and drive gain read back
identical on both backends, the SO-101 gripper closes straight through the vial on PhysX: the moving jaw reaches its
closed limit (-0.155 rad) within 30 control steps while Newton stalls at +0.065 rad; the vial is squeezed out along
the pads regardless of friction (mu = 0.7 or 3.0 make no difference). Contact forces at the pinch: Newton 14-21 N,
PhysX 6 N. Cause: the vial rigid body runs at PhysX's default 4 position iterations and the articulation at the
asset's 8, and the articulation-vs-rigid-body contact does not converge.

**Measured.** Position iterations on the vial and the articulation: 4/8 -> 6 N (fails), 32 -> 12 N (fails),
64 -> 23 N (fails slowly), 128 -> 33 N (holds), 255 -> 31 N (holds). Halving the physics step (240 Hz) alone does
not help. `isaaclab_ov` exposes no solver settings on `OvPhysxCfg` (only GPU capacities and determinism flags), and
`physxScene:solverType` / `physxScene:frictionType` authored on the scene prim are ignored (bit-identical results).

**Workaround here.** `solver_position_iteration_count=128` on the vial (`RigidBodyPropertiesCfg`) and on the
articulation (`ArticulationRootPropertiesCfg`), PhysX preset only (`PHYSX_SOLVER_POSITION_ITERATIONS` in
`env_cfg.py`). Cost: PhysX runs ~18k steps/s at 4,096 envs versus 89k for Newton.

**Fix prompt.**
> In Isaac Lab, `source/isaaclab_ov/isaaclab_ov/physics/ovphysx_cfg.py` (`OvPhysxCfg`) exposes only GPU buffer
> capacities and determinism flags; solver type, friction type, min/max position and velocity iteration counts and the
> friction offset/correlation thresholds cannot be set, and `physxScene:*` attributes authored on the scene prim are not
> honoured by `OvPhysxManager._configure_physx_scene_prim`. (1) Add `solver_type`, `min/max_position_iteration_count`,
> `min/max_velocity_iteration_count`, `friction_type`, `friction_offset_threshold`, `friction_correlation_distance`,
> `bounce_threshold_velocity` and `enable_ccd` to `OvPhysxCfg`, author them on the `PhysicsScene` prim in
> `_configure_physx_scene_prim`, and verify with `PhysX` read-back that the runtime applies them. (2) Make the default
> rigid-body position iteration count used by `isaaclab_ov` match `isaaclab_physx` (`PhysxCfg` defaults) instead of
> the PhysX SDK default of 4, or document the difference in the OV backend docs. (3) Add a regression test that pinches
> a light rigid body between two articulation links with a position drive and asserts the joint stalls (Newton and
> OV PhysX both), using the SO-101 asset if available.

---

## 2. Rigid PhysX contacts drop the vial once the pinch relaxes; Newton's compliant contact keeps it (model gap, no shared config)

**What happens.** After the pinch (issue 1 fixed), both engines relax the pinch while the arm lifts (the gripper
target is the constant close position, yet the jaw drifts open ~0.04 rad on both). Newton keeps the vial hanging on
the 1.3 mm cap ledge because its compliant contact lets the pads sink 1-2 mm into the cap; PhysX's rigid contact goes
to exactly zero force on the moving jaw and the vial slides along the fixed pad and drops (grasp probe: Newton
offset 10 mm at step 59, PhysX 38 mm). mu = 3, mesh cap, positive/negative rest offsets, 255 iterations and velocity
iterations do not fix it. A PhysX compliant contact material with Newton's stiffness/damping (1.57e5 / 1.12e3) does.

**Why this is an Isaac Lab issue.** Newton's contact model in this task is configured through a `NewtonManager`
builder callback (`shape_material_ke/kd`, `mujoco:geom_solref/solimp`), PhysX's through `PhysxMaterialAPI`
compliant-contact attributes. There is no backend-neutral contact-compliance setting, and nothing warns that a
Newton-tuned soft contact will behave as a rigid contact on PhysX.

**Workaround here.** `WORKSHOP_CONTACT_MATERIAL` is the `isaaclab_physx` `RigidBodyMaterialCfg` with
`compliant_contact_stiffness/damping` equal to Newton's ke/kd and `friction_combine_mode="max"` (MuJoCo rule);
Newton ignores the `physxMaterial:*` attributes. Verified on the audit path with a controlled A/B: the PhysX-trained
state checkpoint scores 98.4% with the compliant material and 89.7% (30% unsafe rack contacts) with it stripped.

**Fix prompt.**
> In Isaac Lab, add a backend-neutral contact-compliance description to `RigidBodyMaterialCfg` (stiffness, damping in
> N/m and N s/m) that `isaaclab_newton` maps to `shape_material_ke/kd` (and MuJoCo `solref`) and `isaaclab_physx` /
> `isaaclab_ov` map to `physxMaterial:compliantContactStiffness/Damping`, and make the physics-material randomisation
> events preserve compliant parameters when they rewrite the `(static, dynamic, restitution)` shape material on OV
> PhysX (`randomize_rigid_body_material` writes the 3-value `SHAPE_FRICTION_AND_RESTITUTION` binding). Add a test
> that presses a rigid body between two links with a relaxing drive on both backends and asserts the body stays in
> contact with the same interpenetration budget.

---

## 3. OV PhysX joint friction / viscous friction act weaker than the same values on Newton (semantics / units)

**What happens.** Same joint parameters (read back identical: `DOF_ARMATURE`, `DOF_FRICTION_PROPERTIES`,
`DOF_STIFFNESS`, `DOF_DAMPING`, `DOF_MAX_FORCE`, limits), yet the joints respond faster on PhysX. Open-loop excitation
(folded arm): elbow RMSE 0.143 rad between backends, settles 0.1 rad closer to target on PhysX; fast-sine amplitude
1.087 vs 0.833. Loaded (transport pose): the shoulder moves 2.4x faster per commanded step on PhysX, holding torque
-0.69 vs -1.25 N m at the same pose, no saturation anywhere. Fitted correction factors: friction x1.5-2 on
pan/roll/gripper/shoulder, viscous x1.5-3 on every joint.

**Suspected cause.** `isaaclab_ov.Articulation.write_joint_friction_coefficient_to_sim_index` documents its inputs as
dimensionless coefficients and claims to mirror `isaaclab_physx`; PhysX 5.6's joint friction takes static/dynamic
*efforts* (N m) and a viscous coefficient (N m s / rad), and Newton's `frictionloss` is a torque. The systematic ~2x
on viscous friction across all joints points at a units mismatch in the OV runtime binding or the writer.

**Workaround here.** None shipped: the fitted factors matched the open-loop traces to 0.001-0.005 rad but did not
improve Newton->PhysX transfer and lowered PhysX->Newton transfer (68% -> 42%), so `PHYSX_JOINT_FRICTION_SCALE` /
`PHYSX_JOINT_VISCOUS_SCALE` are 1.0 and the values are only recorded.

**Fix prompt.**
> In Isaac Lab, determine the units and semantics the `ovphysx` `ARTICULATION_DOF_FRICTION_PROPERTIES` binding
> expects for `(static, dynamic, viscous)` (PhysX 5.6 `PxJointFrictionParams` efforts vs the legacy dimensionless
> coefficient) and how the OV runtime maps `physxJoint:*` friction attributes onto it. Make
> `source/isaaclab_ov/isaaclab_ov/assets/articulation/articulation.py::write_joint_friction_coefficient_to_sim_index`
> and the `isaaclab_physx` writer agree and state the units in both docstrings; add a single-joint regression test that
> drives a revolute joint against a known friction effort and checks the steady-state velocity / stall against the
> analytic Coulomb + viscous model (and against `isaaclab_newton` when available); verify with the SO-101 asset that
> the open-loop excitation matches between `presets=newton_mjwarp` and `presets=physx` without correction factors.

---

## 4. OV PhysX contact sensor cannot bind a filtered sensor on a link that shares its name with a joint (bug)

**What happens.** `ContactSensorCfg(prim_path=".../Robot/gripper", filter_prim_paths_expr=[".../Vial"])` fails in
`sim.reset()` with `RuntimeError: Failed to create contact binding: failed to create rigid contact view: no sensor
entries were produced` whenever `num_envs > 1`. The SO-101 asset has a link `gripper` and a joint `gripper`. Isolation:
`Robot/gripper` unfiltered OK; `Robot/moving_jaw_so101_v1` filtered OK; `Vial` filtered to `Rack` + `Robot/wrist` OK;
anything pairing `Robot/gripper` with a filter fails; all fine with `--num_envs 1` and on Newton.

**Workaround here.** Fixed-jaw sensor is unfiltered (`net_forces_w`); the moving-jaw sensor stays filtered to the
vial; `terms._contact_magnitude` picks the right tensor.

**Fix prompt.**
> In Isaac Lab, `source/isaaclab_ov/isaaclab_ov/sensors/contact_sensor/contact_sensor.py` fails to create a filtered
> contact binding for an articulation link whose name is also a joint name in the same articulation when `num_envs > 1`
> (physics-layer clones via `physx.clone()`). Write a regression test next to
> `test_nested_body_contact_sensor_resolution` with a link and a joint sharing a name, replicate with
> `ovphysx_replicate`, create a filtered `ContactSensor` on that link and assert `sensor_count == num_envs` and a correct
> `force_matrix_w` shape. Determine whether `path_expr_to_glob` / `resolve_matching_prims_from_source` also matches
> the joint prim by leaf name (then restrict globs to rigid-body prims and pass exact per-body filter paths) or whether
> the `ovphysx` clone registry keys joints under `<articulation>/<joint>`; in the latter case raise a clear error naming
> the colliding prim and file the runtime issue. Verify with the SO-101 asset at 1, 8 and 1024 envs.

---

## 5. OVRTX cannot compile the MDL materials shipped with the workshop assets and paints them red (renderer limitation)

**What happens.** With `presets=physx,ovrtx` the vial and rack render solid red; the console shows
`[MDLC:COMPILER] error: .../OmniPBR.mdl(21,8): C120 could not find module '.::OmniPBR_ClearCoat' in module path`.
The assets referenced `@OmniPBR.mdl@` relative to the layer (a vendored copy that imports `OmniPBR_ClearCoat`, which
was not vendored), and OVRTX resolves the relative reference before its own shipped library
(`ovrtx/bin/library/mdl/Base/`). The Newton renderer reads `inputs:diffuse_color_constant` from the USD and is fine.
Adding a `UsdPreviewSurface` output *alongside* the MDL output does not help: OVRTX still prefers the MDL context.
Newton-trained camera policies score 0% on PhysX+OVRTX with the red materials.

**Workaround here.** Workshop materials rewritten to `UsdPreviewSurface` only (same colours); the vendored
`OmniPBR.mdl` was removed. Both renderers now draw the scene identically (verified with wrist-camera frames).

**Fix prompt.**
> In Isaac Lab's OVRTX renderer integration (`source/isaaclab_ov/isaaclab_ov/renderers/ovrtx_renderer.py`), when an
> MDL material fails to compile, fall back to the material's universal `outputs:surface` (`UsdPreviewSurface`) if
> present instead of the red error material, and log one warning per material naming the failing module. Also
> document that relatively referenced `.mdl` files must ship their imports, and consider configuring the OVRTX MDL
> search path so that `OmniPBR` and the `nvidia::core_definitions` modules shipped in `ovrtx/bin/library/mdl` resolve
> before a broken local copy. Add a render test with a material carrying both an MDL and a preview-surface output where
> the MDL is unresolvable, asserting the preview-surface colour is rendered.

---

## 6. Vial mass differs between backends because neither backend authors it (asset + code gap)

**What happens.** The vial USD authors no mass or density (only an empty `PhysicsMassAPI` on a disabled mesh prim).
Newton derives 23.26 g, OV PhysX 21.13 g from the same colliders (two cylinders; implied densities 251 vs 228 kg/m^3
do not correspond to any authored value). A 10% mass difference on the manipulated object is a silent sim2sim gap.

**Workaround here.** Not yet applied (would perturb running comparisons); the fix is to author `physics:mass` on the
vial explicitly.

**Fix prompt.**
> In Isaac Lab, when a rigid body has no authored `physics:mass` / `physics:density`, make `isaaclab_newton` and
> `isaaclab_ov` compute the same default mass (same default density, same set of enabled colliders; the disabled
> `collisionEnabled = false` mesh must not contribute), and emit a warning listing bodies whose mass was derived from
> defaults. Add a cross-backend test that spawns an unauthored rigid body and asserts equal `default_mass` on both.

---

## 7. USD `Cylinder` colliders are PhysX custom geometry with poor contact quality on OV PhysX (limitation)

**What happens.** `Cylinder` prims become `PxCustomGeometry` (`ConeCylinderConvexMesh.cpp` / `PhysXCustomGeometry.cpp`).
When the under-converged pinch (issue 1) squeezed the cylinder-collider vial into the mat, it sank through the mat at a
steady 0.13 m/s; the same vial with convex-mesh colliders did not. Newton treats the same prims as analytic cylinders.

**Workaround here.** None needed after issue 1; noted because `Cylinder` colliders are the natural choice for the
tutorial asset.

**Fix prompt.**
> In Isaac Lab's OV PhysX backend, document that `Cylinder`/`Cone` colliders are simulated as PhysX custom geometry
> and expose the `/physics/collisionApproximateCylinders` (convex-mesh approximation) runtime setting on `OvPhysxCfg`
> so users can choose the robust path; add a test that rests a cylinder on a box under a large downward impulse and
> asserts it does not tunnel.

---

## 8. OV PhysX introspection APIs are unusable from Isaac Lab (limitation)

**What happens.** `PhysX.raycast` returns no hits even for a vertical ray into the ground/mat
(`physxScene:enableSceneQuerySupport` is authored `False` by `OvPhysxManager` with no cfg switch);
`PhysX.get_contact_report()` is empty when contact bindings are in use; `PhysX.get_object_type` returns `INVALID` for
every collider prim and for all cloned environments (`env_1..N`). Debugging contact normals, which was essential
here, had to be done indirectly.

**Fix prompt.**
> In Isaac Lab, add `enable_scene_query_support: bool` to `OvPhysxCfg` (default False for speed) and a documented way
> to obtain per-point contact data (`get_contact_report`) alongside contact bindings, and make `get_object_type`
> resolve collider prims and cloned-environment paths; add smoke tests for each.

---

## 9. Cross-backend asset conventions: SO-101 `newton:*` joint attributes have no PhysX counterpart in the asset (asset-level)

**What happens.** The `so101_new_calib_SysID` asset stores the system-identified joint parameters only as
`newton:armature`, `newton:friction` (N m) and `newton:damping` (N m s per **degree**) in its `physics` variant; the
`physx` variant carries no `physxJoint:*` friction/armature values. Anyone switching backends must know to copy the
values into the actuator config and convert damping per degree -> per radian (a factor 57.3 that is easy to miss).

**Workaround here.** `SYS_ID_*` dictionaries in `env_cfg.py` with `math.degrees()` on damping, applied through
`preset(physx=...)`.

**Fix prompt.**
> In the Isaac Lab asset pipeline / `isaaclab_assets/robots/so101.py`, either author the sys-ID values in both USD
> variants (`physxJoint:armature`, static/dynamic/viscous friction attributes for the `physx` variant) or read the
> `newton:*` attributes in the actuator config for all backends with the per-degree -> per-radian conversion done
> once, and document the `newton:damping` units in the Newton backend docs.

---

## 10. Performance note (not a bug): PhysX at the iteration counts the task needs is ~5x slower than Newton

`isaaclab benchmark runtime` at 4,096 envs: Newton 89k steps/s; PhysX 17.8k (compliant material) / 18.5k (rigid).
State-policy training: 34 min on Newton, ~3 h on PhysX. Worth a note in the OV backend docs.

---

## 11. OV PhysX ignores the magnitude of compliant-contact stiffness/damping (only presence matters)

**What happens.** With `physxMaterial:compliantContactStiffness` / `Damping` authored on the workshop material, the
pinch behaves identically for stiffness 1.57e5, 3.14e5, 6.28e5 and 1.256e6 N/m (damping scaled alike): jaw stall
angle +0.018 rad, identical force traces to the tenth of a newton, identical retention. Without the attributes the
contact is rigid (stall +0.099, vial dropped). So the runtime switches between rigid and *some* compliant model but
does not use the authored spring values, which makes it impossible to match Newton's contact stiffness (Newton stalls
at +0.065 with ke = 1.57e5).

**Workaround here.** None; the in-hand equilibrium difference between Newton and PhysX therefore remains (the main
residual Newton->PhysX gap).

**Fix prompt.**
> In Isaac Lab's OV PhysX backend, verify how `physxMaterial:compliantContactStiffness` and
> `compliantContactDamping` authored on a `Material` prim reach the runtime (`PhysxMaterialAPI` parsing in
> `ovphysx`, or `isaaclab_ov` material batches). Author several stiffness values on otherwise identical scenes and assert
> that a body pressed by a position drive penetrates by `F / k`; if the runtime clamps or ignores the values, expose
> the working parameter path (e.g. `compliantContactAccelerationSpring`) or file the runtime issue, and document the
> supported range.
