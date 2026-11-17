"""Active gaze for SO-101 vial placement: a foveated scene camera whose gaze the policy moves.

The retina samples the scene densely where it looks and sparsely in the periphery. A matched non-foveated model is
the same task with a large ``env.scene.eye.cmf_a_deg``. This package needs the ``fovi`` extra, which installs Isaac
Lab's ``isaaclab_fovi`` camera and the policy libraries.
"""

TASK_ID = "IsaacTutorial-Place-Vial-SO101-ActiveGaze"
