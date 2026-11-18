"""Declarative OVRTX shading and renderer presets."""

from isaaclab.utils import configclass
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_ov.renderers import OVRTXRendererCfg

from isaaclab_tasks.utils import PresetCfg


@configclass
class CameraOutputCfg(PresetCfg):
    # Path-traced RGB by default: RTX Minimal mode draws no background or dome light.
    default = ["rgb"]
    newton_renderer = ["rgb"]
    rtpt = ["rgb"]
    minimal_diffuse_mdl = ["simple_shading_diffuse_mdl"]
    minimal_diffuse_mdl_shadows = minimal_diffuse_mdl
    minimal_full_mdl = ["simple_shading_full_mdl"]
    minimal_full_mdl_shadows = minimal_full_mdl


@configclass
class CameraOutputNameCfg(PresetCfg):
    default = "rgb"
    newton_renderer = "rgb"
    rtpt = "rgb"
    minimal_diffuse_mdl = "simple_shading_diffuse_mdl"
    minimal_diffuse_mdl_shadows = minimal_diffuse_mdl
    minimal_full_mdl = "simple_shading_full_mdl"
    minimal_full_mdl_shadows = minimal_full_mdl


@configclass
class ShadowsCfg(PresetCfg):
    # Only Minimal mode reads this switch; path tracing always traces shadows.
    default = False
    rtpt = False
    minimal_diffuse_mdl = False
    minimal_diffuse_mdl_shadows = True
    minimal_full_mdl = False
    minimal_full_mdl_shadows = True


@configclass
class CameraRenderSettingsCfg(PresetCfg):
    """Author native sample settings directly on each camera's render product."""

    newton_renderer = {}
    _minimal = {
        "omni:rtx:post:aa:op": ("Token", "none"),
        "omni:rtx:newDenoiser:enabled": ("Bool", True),
    }
    rtpt = {**_minimal, "omni:rtx:rtpt:spp": ("Int", 4)}
    minimal_diffuse_mdl = _minimal
    minimal_diffuse_mdl_shadows = _minimal
    minimal_full_mdl = _minimal
    minimal_full_mdl_shadows = _minimal
    default = rtpt


@configclass
class FoviRendererCfg(PresetCfg):
    newton_renderer = NewtonWarpRendererCfg(render_order="pixel_priority")
    ovrtx = OVRTXRendererCfg(enable_shadows=ShadowsCfg())
    default = ovrtx
