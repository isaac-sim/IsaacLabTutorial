"""Newton physics manager that applies the workshop contact model to its own simulation."""

from __future__ import annotations

from typing import Any

from isaaclab.physics import PhysicsEvent
from isaaclab_newton.physics import NewtonManager

from .env_cfg import initialize_contacts


class ContactModelNewtonManager(NewtonManager):
    """Writes the workshop contact model when this manager's simulation builds its Newton model.

    Registering on the manager, not process-wide, leaves other simulations in the same process untouched.
    """

    def initialize(self, sim_context: Any) -> None:
        super().initialize(sim_context)
        self.register_callback(self._apply_contact_model, PhysicsEvent.MODEL_INIT, name="so101_workshop_contact_model")

    def _apply_contact_model(self, _event: Any) -> None:
        initialize_contacts(self.builder, self._sim.stage)
