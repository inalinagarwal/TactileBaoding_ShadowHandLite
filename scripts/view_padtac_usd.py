# Task utility: headless RTX render of shadow_padtac.usd from several angles -> PNGs.
import argparse
import os
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.sim.utils import find_matching_prim_paths  # noqa: E402
from pxr import UsdGeom, UsdPhysics, Usd  # noqa: E402

ASSET_DIR = "/home/nalin/roto_2/roto/assets/shadow_lite"
DST = f"{ASSET_DIR}/shadow_padtac.usd"
OUT_DIR = "/home/nalin/roto_2/padtac_views"
os.makedirs(OUT_DIR, exist_ok=True)

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60.0, device="cuda:0"))

# lights
sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)).func("/World/DomeLight", sim_utils.DomeLightCfg(intensity=2500.0))
sim_utils.DistantLightCfg(intensity=2500.0).func("/World/KeyLight", sim_utils.DistantLightCfg(intensity=2500.0), translation=(0.2, -0.3, 0.6))

# spawn the hand (fixed base, no gravity so it holds pose)
robot_cfg = sim_utils.UsdFileCfg(usd_path=DST, activate_contact_sensors=False)
robot_cfg.func("/World/Robot", robot_cfg)

# camera
cam_cfg = CameraCfg(
    prim_path="/World/Camera",
    height=900,
    width=1400,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=28.0, clipping_range=(0.005, 50.0)),
)
camera = Camera(cam_cfg)

sim.reset()

# compute hand centroid from pad + link body world positions
stage = sim.stage
xf = UsdGeom.XformCache()
pts = []
for p in stage.Traverse():
    if p.HasAPI(UsdPhysics.RigidBodyAPI) and str(p.GetPath()).startswith("/World/Robot"):
        t = xf.GetLocalToWorldTransform(p).ExtractTranslation()
        pts.append([t[0], t[1], t[2]])
pts = np.array(pts)
c = pts.mean(0)
lo, hi = pts.min(0), pts.max(0)
rad = float(np.linalg.norm(hi - lo)) * 1.1 + 0.05
print(f"[view] centroid={c}, radius={rad:.3f}, nbodies={len(pts)}")

target = torch.tensor([c], dtype=torch.float32, device=sim.device)

def save_png(arr, path):
    from PIL import Image
    a = arr
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    if a.shape[-1] == 4:
        a = a[..., :3]
    Image.fromarray(a).save(path)
    print("[view] wrote", path)

# a few orbit angles (azimuth deg) at slight elevation
views = {
    "dorsal":  (0.0, 20.0),
    "left":    (60.0, 20.0),
    "right":   (-60.0, 20.0),
    "top":     (0.0, 70.0),
}
for name, (az, el) in views.items():
    a = np.radians(az); e = np.radians(el)
    eye = c + rad * np.array([np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e)])
    eyes = torch.tensor([eye], dtype=torch.float32, device=sim.device)
    camera.set_world_poses_from_view(eyes, target)
    for _ in range(6):
        sim.step()
        camera.update(dt=sim.get_physics_dt())
    rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
    save_png(rgb, f"{OUT_DIR}/padtac_{name}.png")

app.close()
