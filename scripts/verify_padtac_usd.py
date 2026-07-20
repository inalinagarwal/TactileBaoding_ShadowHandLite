# Task utility: inspect shadow_padtac.usd vs source, write report to /tmp file.
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app = AppLauncher(args_cli).app

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

ASSET_DIR = "/home/nalin/roto_2/roto/assets/shadow_lite"
SRC = f"{ASSET_DIR}/shadow_touchlab_col.usd"
DST = f"{ASSET_DIR}/shadow_padtac.usd"
OUT = "/tmp/padtac_verify.txt"

lines = []
def log(s=""):
    lines.append(str(s))

for label, path in [("SOURCE", SRC), ("PADTAC", DST)]:
    log(f"================ {label}: {path} ================")
    stage = Usd.Stage.Open(path)
    dp = stage.GetDefaultPrim()
    log(f"default prim: {dp.GetPath() if dp else None}")
    has_visuals = bool(stage.GetPrimAtPath("/visuals"))
    log(f"has /visuals scope: {has_visuals}")
    rbs, pads, joints = [], [], []
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            rbs.append(p.GetName())
            if p.GetName().startswith("rh_fsr_pad_C"):
                pads.append(p)
        if p.GetTypeName() == "PhysicsFixedJoint":
            joints.append(p)
    log(f"total rigid bodies: {len(rbs)}")
    log(f"pad rigid bodies:   {len(pads)}")
    log(f"fixed joints total: {len(joints)}")
    if pads:
        xf = UsdGeom.XformCache()
        log("pad world positions:")
        for p in sorted(pads, key=lambda x: x.GetName()):
            t = xf.GetLocalToWorldTransform(p).ExtractTranslation()
            col = stage.GetPrimAtPath(p.GetPath().AppendChild("collision"))
            jt = stage.GetPrimAtPath(p.GetPath().AppendChild("fixedjoint"))
            b0 = None
            if jt and jt.GetRelationship("physics:body0").GetTargets():
                b0 = jt.GetRelationship("physics:body0").GetTargets()[0].name
            log(f"  {p.GetName()}  world=({t[0]:+.4f},{t[1]:+.4f},{t[2]:+.4f})  "
                f"collider={'Cylinder' if col and col.GetTypeName()=='Cylinder' else col.GetTypeName() if col else 'NONE'}  "
                f"parent(body0)={b0}")
    log()

with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print("WROTE", OUT)
app.close()
