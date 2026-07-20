# Copyright (c) 2026. Task utility (not part of the training regime).
"""Build shadow_padtac.usd = copy of shadow_touchlab_col.usd + 12 FSR pad prims.

Each pad (rh_fsr_pad_C00..C11) is authored as its OWN rigid body (so a
ContactSensor on rh_fsr_pad_.* reports 12 distinct bodies), with:
  * a thin Cylinder collider (radius 0.004, height 0.0015, axis=Y)  <- matches
    sr_hand_padtac.urdf exactly,
  * a PhysicsFixedJoint locking it to its parent hand link at the URDF origin.

The pad's initial world transform is set to parentWorld * translate(origin) so
it renders in the right place even before physics resolves the joint.

Run (headless):
  /home/nalin/miniforge3/envs/thesis/bin/python scripts/make_padtac_usd.py
"""

import argparse
import shutil

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf  # noqa: E402

ASSET_DIR = "/home/nalin/roto_2/roto/assets/shadow_lite"
SRC = f"{ASSET_DIR}/shadow_touchlab_col.usd"
DST = f"{ASSET_DIR}/shadow_padtac.usd"

RADIUS = 0.004
HEIGHT = 0.0015
PAD_MASS = 1e-3

# (pad_name, parent_link, (x, y, z) in parent frame, sim_channel)  -- from sr_hand_padtac.urdf
PADS = [
    ("rh_fsr_pad_C00", "rh_thproximal", (0.000, -0.011, 0.014), 10),
    ("rh_fsr_pad_C01", "rh_ffproximal", (0.000, -0.009, 0.019), 7),
    ("rh_fsr_pad_C02", "rh_palm",       (0.011, -0.011, 0.082), 4),
    ("rh_fsr_pad_C03", "rh_rfproximal", (0.000, -0.009, 0.019), 9),
    ("rh_fsr_pad_C04", "rh_palm",       (-0.011, -0.011, 0.080), 5),
    ("rh_fsr_pad_C05", "rh_palm",       (0.000, -0.011, 0.064), 2),
    ("rh_fsr_pad_C06", "rh_ffmiddle",   (0.000, -0.009, 0.013), 11),
    ("rh_fsr_pad_C07", "rh_palm",       (0.033, -0.011, 0.080), 3),
    ("rh_fsr_pad_C08", "rh_mfproximal", (0.000, -0.009, 0.019), 8),
    ("rh_fsr_pad_C09", "rh_thmiddle",   (0.000, -0.010, 0.012), 18),
    ("rh_fsr_pad_C10", "rh_mfmiddle",   (0.000, -0.009, 0.013), 12),
    ("rh_fsr_pad_C11", "rh_rfmiddle",   (0.000, -0.009, 0.013), 13),
]


def main() -> None:
    shutil.copy(SRC, DST)
    print(f"[copy] {SRC}\n    -> {DST}")

    stage = Usd.Stage.Open(DST)
    default_prim = stage.GetDefaultPrim()
    print(f"[stage] default prim: {default_prim.GetPath()}")

    # Map link name -> prim path for every rigid body; find the articulation root.
    name2prim = {}
    art_root = None
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI) and art_root is None:
            art_root = prim
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            name2prim[prim.GetName()] = prim
    base = default_prim
    print(f"[stage] articulation root: {art_root.GetPath() if art_root else None}")
    print(f"[stage] rigid-body links found: {len(name2prim)}")

    xf_cache = UsdGeom.XformCache()

    for name, parent, (x, y, z), ch in PADS:
        if parent not in name2prim:
            raise RuntimeError(
                f"parent link {parent!r} not a rigid body in USD. "
                f"available: {sorted(name2prim)}"
            )
        parent_prim = name2prim[parent]
        parent_w = xf_cache.GetLocalToWorldTransform(parent_prim)
        # pad world = translate(origin) applied in parent frame, then parent world
        offset = Gf.Matrix4d().SetTranslate(Gf.Vec3d(x, y, z))
        pad_w = offset * parent_w

        pad_path = base.GetPath().AppendChild(name)
        pad_xform = UsdGeom.Xform.Define(stage, pad_path)
        pad_prim = pad_xform.GetPrim()
        pad_xform.ClearXformOpOrder()
        pad_xform.AddTransformOp().Set(pad_w)

        # rigid body + tiny mass so it is its own contact-sensable link
        UsdPhysics.RigidBodyAPI.Apply(pad_prim)
        mass_api = UsdPhysics.MassAPI.Apply(pad_prim)
        mass_api.CreateMassAttr(PAD_MASS)

        # cylinder collider (also renders -> visible in viewer), red
        col_path = pad_path.AppendChild("collision")
        cyl = UsdGeom.Cylinder.Define(stage, col_path)
        cyl.CreateRadiusAttr(RADIUS)
        cyl.CreateHeightAttr(HEIGHT)
        cyl.CreateAxisAttr(UsdGeom.Tokens.y)
        cyl.CreateExtentAttr([(-RADIUS, -HEIGHT / 2, -RADIUS), (RADIUS, HEIGHT / 2, RADIUS)])
        cyl.GetPrim().CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set(
            [Gf.Vec3f(0.9, 0.1, 0.1)]
        )
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())

        # fixed joint: parent link <-> pad
        j_path = pad_path.AppendChild("fixedjoint")
        joint = UsdPhysics.FixedJoint.Define(stage, j_path)
        joint.CreateBody0Rel().SetTargets([parent_prim.GetPath()])
        joint.CreateBody1Rel().SetTargets([pad_path])
        joint.CreateLocalPos0Attr(Gf.Vec3f(x, y, z))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        print(f"  + {name:16s} parent={parent:14s} ch={ch:<2d} "
              f"world=({pad_w.ExtractTranslation()[0]:+.4f}, "
              f"{pad_w.ExtractTranslation()[1]:+.4f}, {pad_w.ExtractTranslation()[2]:+.4f})")

    stage.GetRootLayer().Save()
    print(f"[save] wrote {DST}")

    # ---- verify pass: reopen and assert 12 pad rigid bodies ----
    stage2 = Usd.Stage.Open(DST)
    pads = [p for p in stage2.Traverse()
            if p.GetName().startswith("rh_fsr_pad_C") and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [p for p in stage2.Traverse() if "fixedjoint" in str(p.GetPath()) and p.GetName() == "fixedjoint"]
    print(f"[verify] pad rigid bodies: {len(pads)} (expect 12)")
    print(f"[verify] pad fixed joints: {len(joints)} (expect 12)")
    assert len(pads) == 12, "expected 12 pad rigid bodies"
    print("[verify] OK")

    simulation_app.close()


if __name__ == "__main__":
    main()
