# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="PhysX LiDAR with 'moving' boxes by despawn/respawn (static colliders only)."
)
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
parser.add_argument(
    "--usd_path",
    type=str,
    default="/localhome/tea21/Desktop/environment_without_people.usd",
    help="USD to spawn per env (optional background)",
)
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

from isaacsim.sensors.physx import _range_sensor  # PhysX LiDAR
from pxr import UsdGeom, Gf, UsdPhysics, Usd, Sdf  # USD APIs

# ---------------- USER EDITS ----------------
REL_LIDAR_PATH = "Environment/turtlebot/turtlebot3_burger/Lidar"
# -------------------------------------------

# Physics settings: keep GPU pipeline ON
settings = carb.settings.get_settings()
settings.set("/physics/use_gpu", True)
settings.set("/physics/cudaDevice", 0)
settings.set("/physics/tensors/device", 0)
settings.set("/physics/use_gpu_pipeline", True)

# ---- Movers / arena params ----
NUM_MOVERS_PER_ENV = 4
BOX_SIZE           = 0.30   # m (edge length)
SPEED_MIN          = 0.6    # m/s
SPEED_MAX          = 1.4    # m/s
JITTER_STD         = 0.25   # rad/s-equivalent heading jitter
ARENA_HALF         = 5.0    # env-local square half-size in XY
WALL_THICK         = 0.05
WALL_HEIGHT        = 0.5
SEED               = 42

# Respawn cadence (every physics step). You can set to >1 to respawn at lower rate.
RESPAWN_EVERY_STEPS = 1

np.random.seed(SEED)

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Optionally load a background USD once per env."""
    my_asset: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Environment",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(usd_path=args_cli.usd_path),
    )

# --------- Utilities ---------
def _normalize_rel(rel_path: str) -> str:
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
    stage = omni.usd.get_context().get_stage()
    hits = []
    for prim in stage.Traverse():
        if substr.lower() in prim.GetName().lower():
            hits.append(prim.GetPath().pathString)
    return hits

def _ensure_xform(path: Sdf.Path):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        prim = stage.DefinePrim(path, "Xform")
    return UsdGeom.Xformable(prim)

# ---- Spawning helpers (STATIC colliders only) ----
def _spawn_static_box(path_parent: str, center_xyz, edge_len: float, color=(0.6, 0.9, 0.9)):
    """
    Create a static collider 'box' at path_parent with a child /geom cube.
    No RigidBody. Collision lives on the /geom and is raycast-visible to PhysX LiDAR.
    """
    stage = omni.usd.get_context().get_stage()

    # parent Xform at pose (no scale on parent)
    xfp = _ensure_xform(Sdf.Path(path_parent))
    ops = {op.GetOpName(): op for op in xfp.GetOrderedXformOps()}
    xf_op = ops.get("xformOp:transform") or xfp.AddTransformOp()
    xfp.SetXformOpOrder([xf_op])
    xf_op.Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(*center_xyz)))

    # child geometry
    geom_path = f"{path_parent}/geom"
    cube = UsdGeom.Cube.Define(stage, Sdf.Path(geom_path))
    cube.CreateSizeAttr(float(edge_len))
    UsdGeom.Gprim(cube.GetPrim()).CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])

    # collider on the geometry (critical for LiDAR)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    cube.GetPrim().CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set("box")
    cube.GetPrim().CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)

    return path_parent

def _remove_prim(path: str):
    stage = omni.usd.get_context().get_stage()
    if prim_exists(path):
        stage.RemovePrim(Sdf.Path(path))

def _env_root_path(i: int) -> str:
    return f"/World/envs/env_{i}"

# ---------------- Per-env state ----------------
# We keep *only* lightweight book-keeping; there are no bodies to update.
_env_paths_last   = {}  # i -> list of prim paths (current boxes) to delete next cycle
_env_pos          = {}  # i -> (N,2) current XY (for our motion model)
_env_vel          = {}  # i -> (N,2) velocities
_env_step_counter = 0

def _build_initial_boxes_for_env(i: int):
    root = _env_root_path(i)
    movers_parent = f"{root}/Movers"

    boxes, pos, vel = [], [], []
    margin  = BOX_SIZE * 0.5 + 0.25
    z_fixed = BOX_SIZE * 0.5 + 0.01
    L = ARENA_HALF

    for k in range(NUM_MOVERS_PER_ENV):
        x = np.random.uniform(-L + margin, L - margin)
        y = np.random.uniform(-L + margin, L - margin)
        path = f"{movers_parent}/box_{k:02d}__v000000"  # we will bump the version
        color = tuple(np.random.uniform(0.25, 0.9, size=3))
        _spawn_static_box(path, [x, y, z_fixed], BOX_SIZE, color=color)

        speed = np.random.uniform(SPEED_MIN, SPEED_MAX)
        theta = np.random.uniform(-np.pi, np.pi)
        pos.append([x, y])
        vel.append([np.cos(theta) * speed, np.sin(theta) * speed])
        boxes.append(path)

    _env_paths_last[i] = boxes
    _env_pos[i]        = np.array(pos, dtype=np.float32)
    _env_vel[i]        = np.array(vel, dtype=np.float32)

def _reflect_inside_square(pos_xy, vel_xy, half_extent, half_box):
    local_min = -(half_extent - half_box)
    local_max = +(half_extent - half_box)

    over_min = pos_xy[:, 0] < local_min
    over_max = pos_xy[:, 0] > local_max
    vel_xy[over_min | over_max, 0] *= -1.0
    pos_xy[over_min, 0] = local_min + (local_min - pos_xy[over_min, 0])
    pos_xy[over_max, 0] = local_max - (pos_xy[over_max, 0] - local_max)

    over_min = pos_xy[:, 1] < local_min
    over_max = pos_xy[:, 1] > local_max
    vel_xy[over_min | over_max, 1] *= -1.0
    pos_xy[over_min, 1] = local_min + (local_min - pos_xy[over_min, 1])
    pos_xy[over_max, 1] = local_max - (pos_xy[over_max, 1] - local_max)

# ---------------- Main loop ----------------
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    stage = omni.usd.get_context().get_stage()
    lidar_if = _range_sensor.acquire_lidar_sensor_interface()

    lidar_paths = build_paths(args_cli.num_envs, REL_LIDAR_PATH)

    # Helpful diagnostics if LiDAR prim isn’t found:
    missing_lidar = [p for p in lidar_paths if not prim_exists(p)]
    if missing_lidar:
        print("[LiDAR] Expected PhysX LiDAR prims not found:")
        for p in missing_lidar:
            print("   -", p)
        cands = list_candidates("lidar")
        if cands:
            print("[LiDAR] Candidates in stage:")
            for c in cands:
                print("   •", c)
            print("[LiDAR] Update REL_LIDAR_PATH to match one of these under /World/envs/env_0/...")
        else:
            print("[LiDAR] No LiDAR prims found. Ensure it's a PhysX LiDAR sensor.")

    # Build after a UI tick so env transforms exist
    omni.kit.app.get_app().update()
    for i in range(args_cli.num_envs):
        _build_initial_boxes_for_env(i)

    # Cook once so initial colliders are live
    scene.write_data_to_sim()
    sim.step()
    print("[INFO]: Setup complete...")

    scan_counter = 0
    z_fixed = BOX_SIZE * 0.5 + 0.01
    half_box = BOX_SIZE * 0.5

    version_counter = 1  # increment every respawn cycle

    while simulation_app.is_running():
        dt = float(sim.get_physics_dt())

        # Update desired positions (pure math) — not touching PhysX yet
        for i in range(args_cli.num_envs):
            vel = _env_vel[i]
            pos = _env_pos[i]

            # heading jitter for variety
            jitter = np.random.randn(len(vel)) * JITTER_STD * dt
            cj, sj = np.cos(jitter), np.sin(jitter)
            vx, vy = vel[:, 0].copy(), vel[:, 1].copy()
            vel[:, 0] = cj * vx - sj * vy
            vel[:, 1] = sj * vx + cj * vy

            # integrate & reflect in software
            pos[:, 0] += vel[:, 0] * dt
            pos[:, 1] += vel[:, 1] * dt
            _reflect_inside_square(pos, vel, ARENA_HALF, half_box)

        # Only every N steps: delete the old static boxes and spawn new ones at new poses
        if scan_counter % RESPAWN_EVERY_STEPS == 0:
            for i in range(args_cli.num_envs):
                # 1) remove previous prims
                for old_path in _env_paths_last[i]:
                    _remove_prim(old_path)

                # 2) spawn new prims at updated positions with **new unique paths**
                new_paths = []
                movers_parent = f"{_env_root_path(i)}/Movers"
                for k in range(NUM_MOVERS_PER_ENV):
                    x, y = float(_env_pos[i][k, 0]), float(_env_pos[i][k, 1])
                    color = tuple(np.random.uniform(0.25, 0.9, size=3))
                    new_path = f"{movers_parent}/box_{k:02d}__v{version_counter:06d}"
                    _spawn_static_box(new_path, [x, y, z_fixed], BOX_SIZE, color=color)
                    new_paths.append(new_path)

                _env_paths_last[i] = new_paths

        # Push stage edits -> PhysX and step once
        scene.write_data_to_sim()
        sim.step()
        scan_counter += 1
        version_counter += 1

        # Optional LiDAR prints (every 10 scans)
        if scan_counter % 10 == 0:
            for p in lidar_paths:
                if not prim_exists(p):
                    continue
                try:
                    ranges = lidar_if.get_linear_depth_data(p)
                    if ranges is None or len(ranges) == 0:
                        print(f"[LiDAR] scan {scan_counter:05d} | {p} | no data")
                    else:
                        print(
                            f"[LiDAR] scan {scan_counter:05d} | {p} | rays={len(ranges)} | "
                            f"min={float(np.min(ranges)):.3f} m | max={float(np.max(ranges)):.3f} m"
                        )
                except Exception as e:
                    print(f"[LiDAR] scan {scan_counter:05d} | {p} | error: {e}")

def main():
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    scene_cfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=40.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    try:
        run_simulator(sim, scene)
    finally:
        # Graceful shutdown
        ctx = omni.usd.get_context()
        if ctx and ctx.get_stage():
            ctx.close_stage()
        try:
            omni.kit.app.get_app().update()
        except Exception:
            pass
        simulation_app.close()

if __name__ == "__main__":
    main()

