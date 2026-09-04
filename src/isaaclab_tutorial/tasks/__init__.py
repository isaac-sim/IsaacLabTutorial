"""Task registrations exposed to the Isaac Lab CLI."""

from isaaclab_tasks.utils import import_packages

import_packages(
    __name__,
    [
        "isaaclab_tutorial.tasks.place_vial.mdp",
        "isaaclab_tutorial.tasks.utils",
    ],
)
