"""Task registrations exposed to the Isaac Lab CLI."""

from isaaclab_tasks.utils import import_packages

import_packages(__name__, ["so101_place_vial.tasks.place_vial.mdp"])
