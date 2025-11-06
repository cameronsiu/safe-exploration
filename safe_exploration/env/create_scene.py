# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Interactive scene with PhysX LiDAR + IMU printing (multi-env).")
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

# IMU wrapper (physics IMU). If unavailable in your version, we still run (IMU skipped).
try:
    from isaacsim.sensors.physics import IMUSensor
    HAS_IMU_WRAPPER = True
except Exception as _e:
    carb.log_warn(f"[IMU] IMU wrapper not available: {_e}")
    HAS_IMU_WRAPPER = False

# ---------------- USER EDITS ----------------
# These are relative to each env namespace, i.e., resolved as:
# /World/envs/env_0/<REL_*_PATH>, /World/envs/env_1/<REL_*_PATH>, ...
REL_LIDAR_PATH = "Environment/turtlebot/turtlebot3_burger/Lidar"       # <-- edit to match your stage
REL_IMU_PATH   = "Environment/turtlebot/turtlebot3_burger/Imu_Sensor"  # <-- edit or leave empty "" if none
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
        prim_path="{ENV_REGEX_NS}/Environment",   # <-- THIS is the correct way for multi-env
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(usd_path=args_cli.usd_path),
    )

# ---------------- Path helpers ----------------
def _normalize_rel(rel_path: str) -> str:
    """Ensure sensor paths are relative to env root (strip any accidental /World/...)."""
    if not rel_path:
        return rel_path
    rel = rel_path.lstrip("/")  # strip leading slash
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

# ---------------- Main loop ----------------
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Interfaces/wrappers
    lidar_if = _range_sensor.acquire_lidar_sensor_interface()

    # Build per-env sensor paths (relative → absolute under each env)
    lidar_paths = build_paths(args_cli.num_envs, REL_LIDAR_PATH)
    imu_paths   = build_paths(args_cli.num_envs, REL_IMU_PATH) if REL_IMU_PATH else []

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
            print("[LiDAR] No 'lidar' candidates found. Ensure your USD contains a **PhysX** LiDAR prim.")

    # Bind IMU wrappers (best effort)
    imus = {}
    if HAS_IMU_WRAPPER and imu_paths:
        missing_imu = [p for p in imu_paths if not prim_exists(p)]
        if missing_imu:
            print("[IMU] Expected IMU prims not found:")
            for p in missing_imu:
                print("   -", p)
            cands = list_candidates("imu")
            if cands:
                print("[IMU] Candidates found in stage:")
                for c in cands:
                    print("   •", c)
                print("[IMU] Set REL_IMU_PATH so /World/envs/env_0/<REL_IMU_PATH> matches one of these.")

        for p in imu_paths:
            if prim_exists(p):
                try:
                    # frequency=None → use sim tick; read_gravity set at query
                    imus[p] = IMUSensor(prim_path=p, name=f"imu_{p.split('/')[-2]}", frequency=None)
                except Exception as e:
                    carb.log_warn(f"[IMU] Failed to wrap IMU at {p}: {e}")

    scan_counter = 0
    step_counter = 0

    while simulation_app.is_running():
        # Write & step
        scene.write_data_to_sim()
        sim.step()
        scan_counter += 1
        step_counter += 1

        # ------------ Print every 10 scans ------------
        if scan_counter % 10 == 0:
            # LiDAR summary
            for p in lidar_paths:
                if not prim_exists(p):
                    continue  # avoid "Sensor does not exist"
                try:
                    ranges = lidar_if.get_linear_depth_data(p)  # numpy array of meters
                    if ranges is None or len(ranges) == 0:
                        print(f"[LiDAR] scan {scan_counter:05d} | {p} | no data")
                    else:
                        # print(f"[LiDAR] scan {scan_counter:05d} | {p} | rays={len(ranges)} | "
                        #       f"min={float(np.min(ranges)):.3f} m | max={float(np.max(ranges)):.3f} m")
                        print(ranges)
                        # To dump full array each time, uncomment:
                        # print(ranges)
                except Exception as e:
                    print(f"[LiDAR] scan {scan_counter:05d} | {p} | error: {e}")

            # IMU summary (if available)
            if imus:
                for p, imu in imus.items():
                    try:
                        frame = imu.get_current_frame(read_gravity=True)  # includes gravity in lin_acc
                        if not frame:
                            print(f"[IMU]   scan {scan_counter:05d} | {p} | no data")
                            continue
                        # la = np.asarray(frame.get("lin_acc", [0, 0, 0]), dtype=float)
                        # av = np.asarray(frame.get("ang_vel", [0, 0, 0]), dtype=float)
                        # ori = np.asarray(frame.get("orientation", [0, 0, 0, 1]), dtype=float)
                        # print(
                        #     f"[IMU]   scan {scan_counter:05d} | {p} | "
                        #     f"lin_acc=({la[0]:+.3f},{la[1]:+.3f},{la[2]:+.3f}) m/s^2 | "
                        #     f"ang_vel=({av[0]:+.3f},{av[1]:+.3f},{av[2]:+.3f}) rad/s | "
                        #     f"q=({ori[0]:+.3f},{ori[1]:+.3f},{ori[2]:+.3f},{ori[3]:+.3f})"
                        # )
                        print(frame)
                    except Exception as e:
                        print(f"[IMU]   scan {scan_counter:05d} | {p} | error: {e}")

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
