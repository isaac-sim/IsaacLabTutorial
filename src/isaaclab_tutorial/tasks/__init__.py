"""Task registrations exposed to the Isaac Lab CLI."""

from isaaclab_tasks.utils import import_packages

# Discover task registrations without importing the bundled reference tasks or their simulator dependencies.
import_packages(
    __name__,
    [
        "isaaclab_tutorial.tasks.place_vial.mdp",
        "isaaclab_tutorial.tasks.velocity.config.microduck.compat",
    ],
)
