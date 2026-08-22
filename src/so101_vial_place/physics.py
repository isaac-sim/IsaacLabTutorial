# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Newton contact initialization validated by the SO-101 workshop.

The identified joint drives remain authored entirely in the robot USD.  This
module configures the solver's shape contact model, which is separate from the
robot actuators and must be initialized on Newton's completed model builder.
"""

from __future__ import annotations

import time

import newton
import numpy as np
from isaaclab.physics import PhysicsEvent
from isaaclab_newton.physics import NewtonManager

_CONTACT_STIFFNESS = 1.57e5
_CONTACT_DAMPING = 1.12e3
_FRICTION = 0.7
_ROLLING_FRICTION = 0.05
_TORSIONAL_FRICTION = 0.005
_SOLIMP = (0.7, 0.95, 0.0001, 0.5, 2.0)
_SOLREF = (0.002, 1.5)

_registered = False
_sdf_registered = False
_saved_rack_meshes: dict[str, newton.Mesh] = {}


def _repair_rack_mesh(mesh: newton.Mesh, label: str) -> newton.Mesh:
    """Return the workshop rack as a watertight mesh for signed distance."""
    import trimesh

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3)
    repaired = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not repaired.is_watertight:
        repaired.process(validate=True)
        trimesh.repair.fix_normals(repaired)
        trimesh.repair.fix_winding(repaired)
        trimesh.repair.fill_holes(repaired)
    if not repaired.is_watertight:
        raise RuntimeError(f"The workshop rack collider {label!r} is not watertight after repair.")
    return newton.Mesh(
        np.asarray(repaired.vertices, dtype=np.float32),
        np.asarray(repaired.faces, dtype=np.int32).reshape(-1),
    )


def _initialize_rack_sdf(_event: PhysicsEvent) -> None:
    """Restore and build the detailed four-hole rack after Newton cloning."""
    builder = NewtonManager._builder
    if builder is None:
        return
    built = 0
    for shape_index in range(len(builder.shape_body)):
        label = builder.shape_label[shape_index] if shape_index < len(builder.shape_label) else ""
        mesh = _saved_rack_meshes.get(label)
        if mesh is None:
            continue
        scale = np.asarray(builder.shape_scale[shape_index], dtype=np.float32)
        if not np.allclose(scale, 1.0):
            mesh = mesh.copy(vertices=mesh.vertices * scale, recompute_inertia=True)
            builder.shape_scale[shape_index] = (1.0, 1.0, 1.0)
        builder.shape_source[shape_index] = mesh
        builder.shape_type[shape_index] = newton.GeoType.MESH
        start = time.time()
        mesh.build_sdf(
            max_resolution=128,
            narrow_band_range=(-0.003, 0.003),
            margin=0.003,
        )
        print(f"[RACK SDF] {label}: built detailed four-hole collider in {time.time() - start:.2f}s", flush=True)
        built += 1
    if built < 1:
        rack_labels = [label for label in builder.shape_label if "rack" in label.lower()]
        raise RuntimeError(f"The workshop rack collider was not imported; rack-shaped labels: {rack_labels!r}.")
    _saved_rack_meshes.clear()


def _register_rack_sdf() -> None:
    """Preserve the source rack mesh when Newton convexifies other meshes."""
    global _sdf_registered
    if _sdf_registered:
        return
    original_approximate = newton.ModelBuilder.approximate_meshes

    def approximate_with_rack_preserved(
        builder,
        method="convex_hull",
        shape_indices=None,
        raise_on_failure=False,
        keep_visual_shapes=False,
        **kwargs,
    ):
        for shape_index in range(len(builder.shape_body)):
            if builder.shape_type[shape_index] != newton.GeoType.MESH:
                continue
            label = builder.shape_label[shape_index] if shape_index < len(builder.shape_label) else ""
            if "Rack" not in label or "Body1" not in label:
                continue
            if label not in _saved_rack_meshes and builder.shape_source[shape_index] is not None:
                _saved_rack_meshes[label] = _repair_rack_mesh(builder.shape_source[shape_index].copy(), label)
        return original_approximate(
            builder,
            method,
            shape_indices=shape_indices,
            raise_on_failure=raise_on_failure,
            keep_visual_shapes=keep_visual_shapes,
            **kwargs,
        )

    newton.ModelBuilder.approximate_meshes = approximate_with_rack_preserved
    NewtonManager.register_callback(_initialize_rack_sdf, PhysicsEvent.MODEL_INIT, name="workshop_rack_sdf")
    _sdf_registered = True


def _initialize_contacts(_event: PhysicsEvent) -> None:
    builder = NewtonManager._builder
    if builder is None:
        return

    num_shapes = len(builder.shape_body)
    for shape_index in range(num_shapes):
        builder.shape_material_ke[shape_index] = _CONTACT_STIFFNESS
        builder.shape_material_kd[shape_index] = _CONTACT_DAMPING
        builder.shape_material_mu[shape_index] = _FRICTION
        builder.shape_material_mu_rolling[shape_index] = _ROLLING_FRICTION
        builder.shape_material_mu_torsional[shape_index] = _TORSIONAL_FRICTION

    # Prototype builders register these attributes, but Newton's cloner does
    # not currently carry that registration to the main builder.
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    for name, value in (("mujoco:geom_solimp", _SOLIMP), ("mujoco:geom_solref", _SOLREF)):
        attribute = builder.custom_attributes.get(name)
        if attribute is None:
            continue
        if attribute.values is None:
            attribute.values = {}
        for shape_index in range(num_shapes):
            attribute.values[shape_index] = value


def register_so101_contact_model() -> None:
    """Register the workshop-validated shape contact model exactly once."""
    global _registered
    if _registered:
        return
    _register_rack_sdf()
    NewtonManager.register_callback(
        _initialize_contacts,
        PhysicsEvent.MODEL_INIT,
        name="so101_workshop_contact_model",
    )
    _registered = True
