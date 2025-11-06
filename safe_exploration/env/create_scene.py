# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Interactive scene with PhysX LiDAR + GT pose/velocity (multi-env).")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
parser.add_argument("--usd_path", type=str, default="/localhome/tea21/Desktop/environment_without_people.usd", help="USD to spawn per env")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------- Imports ----------------
import numpy as np
import carb
import omni

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.sim.spawners.from_files import UsdFileCfg

from isaacsim.sensors.physx import _range_sensor  # PhysX LiDAR interface
from pxr import UsdGeom, Gf, UsdPhysics, Usd      # for GT pose/velocity

# ---------------- USER EDITS ----------------
# These are relative to each env namespace, resolved as:
# /World/envs/env_0/<REL_*_PATH>, /World/envs/env_1/<REL_*_PATH>, ...
REL_LIDAR_PATH = "Environment/turtlebot/turtlebot3_burger/Lidar"            # <-- edit to match your stage
REL_BASE_PATH  = "Environment/turtlebot/turtlebot3_burger/base_footprint"   # <-- set to your TurtleBot rigid body prim (base_link/base_footprint)
# -------------------------------------------

# Optional: physics GPU settings
my_setting = carb.settings.get_settings()
my_setting.set("/physics/use_gpu", True)
my_setting.set("/physics/cudaDevice", 0)
my_setting.set("/physics/tensors/device", 0)
my_setting.set("/physics/use_gpu_pipeline", True)

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Loads your USD asset once per env (USES ENV REGEX TOKEN!)."""
    my_asset: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Environment",   # <-- correct pattern for multi-env
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(usd_path=args_cli.usd_path),
    )

# ---------------- Path helpers ----------------
def _normalize_rel(rel_path: str) -> str:
    """Ensure sensor/rigid-body paths are relative to env root (strip leading /World/...)."""
    if not rel_path:
        return rel_path
    rel = rel_path.lstrip("/")
    if rel.startswith("World/"):
        rel = rel[len("World/"):]
    return rel

def build_paths(num_envs: int, rel_path: str):
    rel = _normalize_rel(rel_path)
    return [f"/World/envs/env_{i}/{rel}" for i in range(num_envs)]

def prim_exists(path: str) -> bool:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)
    return bool(prim and prim.IsValid())

def list_candidates(substr: str):
    """List prim paths whose name contains 'substr' (case-insensitive)."""
    stage = omni.usd.get_context().get_stage()
    hits = []
    for prim in stage.Traverse():
        if substr.lower() in prim.GetName().lower():
            hits.append(prim.GetPath().pathString)
    return hits

# --- ground-truth readers ---
def get_world_pose(stage, path: str):
    """Return (pos_xyz, quat_xyzw) world pose for the given prim path."""
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return None, None
    xf = UsdGeom.Xformable(prim)
    # World transform at default time; updated after sim.step()
    wt = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = wt.ExtractTranslation()
    q = wt.ExtractRotation().GetQuaternion()  # Gf.Quatd: (imag xyz, real w)
    pos = np.array([t[0], t[1], t[2]], dtype=float)
    quat_xyzw = np.array([q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2], q.GetReal()], dtype=float)
    return pos, quat_xyzw

def get_world_vel(stage, path: str):
    """Return (lin_vel_xyz, ang_vel_xyz) from PhysX RigidBodyAPI (world frame)."""
    rb = UsdPhysics.RigidBodyAPI.Get(stage, path)
    if not rb:
        return None, None
    lin = rb.GetVelocityAttr().Get()            # tuple or Gf.Vec3f
    ang = rb.GetAngularVelocityAttr().Get()
    if lin is None or ang is None:
        return None, None
    lin = np.array([lin[0], lin[1], lin[2]], dtype=float)
    ang = np.array([ang[0], ang[1], ang[2]], dtype=float)
    return lin, ang

# ---------------- Main loop ----------------
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    stage = omni.usd.get_context().get_stage()

    # PhysX LiDAR interface
    lidar_if = _range_sensor.acquire_lidar_sensor_interface()

    # Build per-env paths
    lidar_paths = build_paths(args_cli.num_envs, REL_LIDAR_PATH)
    base_paths  = build_paths(args_cli.num_envs, REL_BASE_PATH)

    # Verify prim existence and help user adjust if needed
    missing_lidar = [p for p in lidar_paths if not prim_exists(p)]
    if missing_lidar:
        print("[LiDAR] Expected LiDAR prims not found:")
        for p in missing_lidar:
            print("   -", p)
        cands = list_candidates("lidar")
        if cands:
            print("[LiDAR] Candidates found in stage:")
            for c in cands:
                print("   •", c)
            print("[LiDAR] Set REL_LIDAR_PATH so /World/envs/env_0/<REL_LIDAR_PATH> matches one of these.")
        else:
            print("[LiDAR] No 'lidar' candidates found. Ensure your USD contains a PhysX LiDAR prim.")

    missing_base = [p for p in base_paths if not prim_exists(p)]
    if missing_base:
        print("[GT] Expected TurtleBot base prims not found:")
        for p in missing_base:
            print("   -", p)
        print("[GT] Try candidates containing 'base' or 'link':")
        for c in list_candidates("base"):
            print("   •", c)
        for c in list_candidates("link"):
            print("   •", c)
        print("[GT] Set REL_BASE_PATH so /World/envs/env_0/<REL_BASE_PATH> points to the robot's rigid body (base_link/base_footprint).")

    scan_counter = 0
    step_counter = 0

    while simulation_app.is_running():
        # Write & step
        scene.write_data_to_sim()
        sim.step()
        scan_counter += 1
        step_counter += 1

        # ------------ Print every 10 steps ------------
        if scan_counter % 10 == 0:
            # LiDAR summary (optional: swap to print full array)
            for p in lidar_paths:
                if not prim_exists(p):
                    continue
                try:
                    ranges = lidar_if.get_linear_depth_data(p)
                    if ranges is None or len(ranges) == 0:
                        print(f"[LiDAR] scan {scan_counter:05d} | {p} | no data")
                    else:
                        print(f"[LiDAR] scan {scan_counter:05d} | {p} | rays={len(ranges)} | "
                              f"min={float(np.min(ranges)):.3f} m | max={float(np.max(ranges)):.3f} m")
                        # print(ranges)  # uncomment to dump full array
                except Exception as e:
                    print(f"[LiDAR] scan {scan_counter:05d} | {p} | error: {e}")

            # Ground-truth pose & velocity
            for p in base_paths:
                if not prim_exists(p):
                    continue
                pos, quat = get_world_pose(stage, p)
                lin, ang  = get_world_vel(stage, p)
                if pos is None or lin is None:
                    print(f"[GT]    scan {scan_counter:05d} | {p} | no data (check that prim is a RigidBody)")
                else:
                    print(
                        f"[GT]    scan {scan_counter:05d} | {p} | "
                        f"pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) m | "
                        f"lin_vel=({lin[0]:+.3f},{lin[1]:+.3f},{lin[2]:+.3f}) m/s | "
                        f"ang_vel=({ang[0]:+.3f},{ang[1]:+.3f},{ang[2]:+.3f}) rad/s | "
                        f"quat=({quat[0]:+.3f},{quat[1]:+.3f},{quat[2]:+.3f},{quat[3]:+.3f})"
                    )

        # Optional: periodic scene reset
        if step_counter % 500 == 0:
            step_counter = 0
            scene.reset()
            print("[INFO]: Resetting scene...")

def main():
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0")
    sim = SimulationContext(sim_cfg)

    # Camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    # Build scene with multiple envs; spacing so envs don't collide
    scene_cfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=40.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")

    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()

