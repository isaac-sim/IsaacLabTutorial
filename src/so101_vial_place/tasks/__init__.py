"""Task registrations exposed to the Isaac Lab CLI."""

from isaaclab_tasks.utils import import_packages

import_packages(__name__, [".mdp", "utils"])
