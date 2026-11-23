"""Task registrations exposed to the Isaac Lab CLI."""

from isaaclab_tasks.utils import import_packages

# Discover task registrations without importing simulator dependencies.
import_packages(
    __name__,
    [
        "isaaclab_tutorial.tasks.place_vial.mdp",
        "isaaclab_tutorial.tasks.velocity.config.microduck.bam",
        "isaaclab_tutorial.tasks.velocity.mdp",
    ],
)
