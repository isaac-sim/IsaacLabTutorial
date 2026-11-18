"""Project-owned BAM support for the unmodified Isaac Lab 3.0.0 package.

Call install before constructing an environment. The hooks add support only for
BamActuatorCfg; GA's standard actuators retain their original implementation.
"""

from importlib.metadata import version

_installed = False


def install():
    global _installed
    if _installed:
        return
    if version('isaaclab') != '3.0.0':
        raise RuntimeError('This compatibility implementation requires Isaac Lab 3.0.0 GA.')
    from .config import install_config_resolution
    from .integration import install_hooks
    install_hooks()
    install_config_resolution()
    _installed = True
