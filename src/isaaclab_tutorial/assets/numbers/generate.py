"""Generate four static, reusable number decals; consumers set their physical size.

Each numeral has its own texture and USD asset, with no runtime material writes.
No rigid body or collider is authored. Z is the surface normal, Y is text-up.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

CELL = 256
FONT_SIZE = 260


def generate(output: Path, font: Path):
    output.mkdir(parents=True, exist_ok=True)
    face = ImageFont.truetype(str(font), FONT_SIZE)
    for number in range(1, 5):
        image = Image.new("RGB", (CELL, CELL), "white")
        draw = ImageDraw.Draw(image)
        text = str(number)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=face)
        draw.text(
            ((CELL - right + left) / 2 - left, (CELL - bottom + top) / 2 - top),
            text,
            font=face,
            fill="black",
        )
        image.save(output / f"number_{number}.png")
        generate_label(output, number)
    generate_review(output)


def generate_label(output, number):
    stage = Usd.Stage.CreateNew(str(output / f"number_{number}.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    root = UsdGeom.Xform.Define(stage, "/NumberLabel")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/NumberLabel/Decal")
    mesh.CreatePointsAttr([(-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(False)
    uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex")
    uv.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
    material = UsdShade.Material.Define(stage, "/NumberLabel/Looks/Ink")
    shader = UsdShade.Shader.Define(stage, "/NumberLabel/Looks/Ink/Shader")
    shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    out.SetRenderType("material")
    material.CreateSurfaceOutput("mdl").ConnectToSource(out)
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f"number_{number}.png"))
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set((1, 1, 1))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.85)
    shader.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(0.05)
    shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(False)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()

    check = Usd.Stage.Open(str(output / f"number_{number}.usda"))
    assert check.GetDefaultPrim().GetPath() == "/NumberLabel"
    bounds = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
        .ComputeWorldBound(check.GetDefaultPrim())
        .ComputeAlignedRange()
    )
    assert bounds.GetSize() == Gf.Vec3d(1, 1, 0)
    assert not any(p.HasAPI(UsdPhysics.CollisionAPI) for p in check.Traverse())


def generate_review(output):
    # Reusable-reference check, plus a review stage with all four static assets.
    for name, count in (("preview", 1), ("review", 4)):
        review = Usd.Stage.CreateNew(str(output / f"{name}.usda"))
        UsdGeom.SetStageUpAxis(review, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(review, 1)
        scene = UsdGeom.Xform.Define(review, "/Review")
        review.SetDefaultPrim(scene.GetPrim())
        for i in range(count):
            label = UsdGeom.Xform.Define(review, f"/Review/Label{i}")
            label.GetPrim().GetReferences().AddReference(f"number_{i + 1}.usda")
            label.AddTranslateOp().Set((i * 0.03, 0, 0))
            label.AddScaleOp().Set((0.022, 0.022, 0.022))
        light = UsdLux.DomeLight.Define(review, "/Review/Light")
        light.CreateIntensityAttr(1500)
        camera = UsdGeom.Camera.Define(review, "/Review/Camera")
        camera.AddTranslateOp().Set(((count - 1) * 0.015, 0, 0.15))
        camera.CreateClippingRangeAttr((0.001, 10))
        review.GetRootLayer().Save()
    print(f"Validated {output}: four static unit decals, relative textures, no colliders")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, default=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
    args = parser.parse_args()
    generate(args.output, args.font)
