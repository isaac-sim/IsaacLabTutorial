# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Newton contact initialization validated by the SO-101 workshop.

The identified joint drives remain authored entirely in the robot USD.  This
module configures the solver's shape contact model, which is separate from the
robot actuators and must be initialized on Newton's completed model builder.
"""

from __future__ import annotations

import newton
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
    NewtonManager.register_callback(
        _initialize_contacts,
        PhysicsEvent.MODEL_INIT,
        name="so101_workshop_contact_model",
    )
    _registered = True
