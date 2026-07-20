#!/usr/bin/env python
"""Build shadow_padtac_biotac.usd = shadow_padtac.usd (12 FSR pads) with the 4
distal fingertips' geometry SWAPPED from the TouchLab tip (fingertip_v5_simple)
to the official Shadow BioTac mesh.

Rationale (16-sensor tactile parity with hardware):
  * The real hand has BioTac fingertips; sim had TouchLab tips. The policy senses
    contact on the 4 distal links -> deploy channels 15/16/17/22 (ff/mf/rf/th dist).
  * We use the ENTIRE BioTac mesh (convex-hull collider) as the contact geometry -
    no cylinder proxy, so there is no sensing gap at the fingertip.
  * The 12 FSR pad bodies from shadow_padtac.usd are left untouched.

Per distal link /shadowhand_motor/rh_{ff,mf,rf,th}distal we:
  1. deactivate the existing TouchLab `visuals` + `collisions` children,
  2. add `bt_visual` (full BioTac mesh, biotac-green, render only),
  3. add `bt_collision` (BioTac convex hull + CollisionAPI + MeshCollisionAPI=convexHull).

The BioTac .dae (scaled x0.001 -> m) is authored in the distal-link frame already
(z in [0.003, 0.035] m), matching where the TouchLab collider tip sat (~0.0345 m),
so no extra offset is applied.

Run (standalone pxr; no Isaac app needed):
  EXT=.../omni.usd.libs-.../ ; LD_LIBRARY_PATH=$EXT/bin:$CONDA/lib \
  PYTHONPATH=$EXT python scripts/make_padtac_biotac_usd.py
"""

import shutil

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

ASSET_DIR = "/home/nalin/roto_2/roto/assets/shadow_lite"
SRC = f"{ASSET_DIR}/shadow_padtac.usd"
DST = f"{ASSET_DIR}/shadow_padtac_biotac.usd"

MESH_MM_TO_M = 0.001  # .dae is in millimetres; URDF references it with scale 0.001

# BioTac SP meshes (official Shadow description). Switch *_sp -> *_2p for BioTac-2P.
BIOTAC_F_DAE = f"{ASSET_DIR}/meshes/components/f_distal/bt_sp/f_distal_bt_sp.dae"
BIOTAC_TH_DAE = f"{ASSET_DIR}/meshes/components/th_distal/bt_sp/th_distal_bt_sp.dae"

BIOTAC_GREEN = (0.16, 0.55, 0.30)

# distal link name -> which biotac mesh ("f" finger or "th" thumb)
DISTALS = [
    ("rh_ffdistal", "f"),
    ("rh_mfdistal", "f"),
    ("rh_rfdistal", "f"),
    ("rh_thdistal", "th"),
]


def load_dae(path: str):
    """Return (verts_m Nx3 float32, faces Mx3 int, hull_verts, hull_faces)."""
    m = trimesh.load(path, force="mesh")
    verts = np.asarray(m.vertices, dtype=np.float64) * MESH_MM_TO_M
    faces = np.asarray(m.faces, dtype=np.int64)
    hull = m.convex_hull
    hverts = np.asarray(hull.vertices, dtype=np.float64) * MESH_MM_TO_M
    hfaces = np.asarray(hull.faces, dtype=np.int64)
    print(
        f"  [{path.split('/')[-1]}] verts={len(verts)} faces={len(faces)} "
        f"hull_verts={len(hverts)} hull_faces={len(hfaces)} "
        f"bbox(m)={verts.min(0).round(4)}..{verts.max(0).round(4)}"
    )
    return verts, faces, hverts, hfaces


def author_mesh(stage, path, verts, faces, color, collision):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([int(i) for i in faces.reshape(-1)])
    mn, mx = verts.min(0), verts.max(0)
    mesh.CreateExtentAttr(
        [Gf.Vec3f(float(mn[0]), float(mn[1]), float(mn[2])),
         Gf.Vec3f(float(mx[0]), float(mx[1]), float(mx[2]))]
    )
    mesh.GetPrim().CreateAttribute(
        "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray
    ).Set([Gf.Vec3f(*color)])
    if collision:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        mca = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        mca.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    return mesh


def main() -> None:
    shutil.copy(SRC, DST)
    print(f"[copy] {SRC}\n    -> {DST}")

    print("[load] BioTac meshes:")
    f_v, f_f, f_hv, f_hf = load_dae(BIOTAC_F_DAE)
    th_v, th_f, th_hv, th_hf = load_dae(BIOTAC_TH_DAE)

    stage = Usd.Stage.Open(DST)

    # Shared BioTac geometry (authored once, referenced by the distal links).
    UsdGeom.Scope.Define(stage, "/bt_geom")
    author_mesh(stage, "/bt_geom/f_visual", f_v, f_f, BIOTAC_GREEN, collision=False)
    author_mesh(stage, "/bt_geom/th_visual", th_v, th_f, BIOTAC_GREEN, collision=False)
    author_mesh(stage, "/bt_geom/f_col", f_hv, f_hf, BIOTAC_GREEN, collision=True)
    author_mesh(stage, "/bt_geom/th_col", th_hv, th_hf, BIOTAC_GREEN, collision=True)
    print("[author] /bt_geom f/th visual+col meshes")

    for distal, kind in DISTALS:
        base = f"/shadowhand_motor/{distal}"
        prim = stage.GetPrimAtPath(base)
        if not prim:
            raise RuntimeError(f"distal link {base} not found")

        # 1. deactivate TouchLab visuals + collisions
        for child in ("visuals", "collisions"):
            cp = stage.GetPrimAtPath(f"{base}/{child}")
            if cp:
                cp.SetActive(False)

        # 2. BioTac visual (internal ref to shared mesh)
        vis = UsdGeom.Mesh.Define(stage, f"{base}/bt_visual")
        vis.GetPrim().GetReferences().AddInternalReference(f"/bt_geom/{kind}_visual")

        # 3. BioTac collision (internal ref to shared convex-hull mesh)
        col = UsdGeom.Mesh.Define(stage, f"{base}/bt_collision")
        col.GetPrim().GetReferences().AddInternalReference(f"/bt_geom/{kind}_col")

        print(f"  ~ {distal:14s} -> BioTac ({kind}) visual+collision; TouchLab deactivated")

    stage.GetRootLayer().Save()
    print(f"[save] wrote {DST}")

    # ---- verify pass ----
    s2 = Usd.Stage.Open(DST)
    pads = [p for p in s2.Traverse()
            if p.GetName().startswith("rh_fsr_pad_C") and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    print(f"[verify] FSR pad bodies: {len(pads)} (expect 12)")
    assert len(pads) == 12, "expected 12 FSR pad bodies preserved"

    for distal, _ in DISTALS:
        base = f"/shadowhand_motor/{distal}"
        assert s2.GetPrimAtPath(f"{base}/bt_collision"), f"missing bt_collision on {distal}"
        old = s2.GetPrimAtPath(f"{base}/collisions")
        assert old and not old.IsActive(), f"old TouchLab collider still active on {distal}"
        col = s2.GetPrimAtPath(f"{base}/bt_collision")
        assert col.HasAPI(UsdPhysics.CollisionAPI), f"bt_collision missing CollisionAPI on {distal}"
    print(f"[verify] 4 distal tips swapped to BioTac + collision OK")
    print("[verify] OK")


if __name__ == "__main__":
    main()
