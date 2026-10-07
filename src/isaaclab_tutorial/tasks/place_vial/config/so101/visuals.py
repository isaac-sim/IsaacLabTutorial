"""Shared camera appearance for the workshop and its recording views."""

from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

CAMERA_BACKGROUND_COLOR = (0.3, 0.3, 0.3)
"""Neutral gray background; native renderer output transforms may differ."""


def workshop_camera_renderer_cfg() -> MultiBackendRendererCfg:
    """Enable the supported shadow settings on both standalone renderers."""
    cfg = MultiBackendRendererCfg()
    cfg.newton_renderer.enable_shadows = True
    cfg.default.enable_shadows = True
    cfg.ovrtx.enable_shadows = True
    return cfg
