"""SO-101 MDP terms, loaded lazily to keep pure helpers simulator-independent."""

from importlib import import_module


def __getattr__(name: str):
    """Resolve task terms first, then Isaac Lab's standard MDP terms."""
    for module_name in (".terms", ".events"):
        task_module = import_module(module_name, __name__)
        if hasattr(task_module, name):
            return getattr(task_module, name)
    upstream_terms = import_module("isaaclab.envs.mdp")
    try:
        return getattr(upstream_terms, name)
    except AttributeError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
