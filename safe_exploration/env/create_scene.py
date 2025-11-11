# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Multi-env + PhysX LiDAR + kinematic moving boxes (GPU pipeline safe, kinematic targets)."
)
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
parser.add_argument(
    "--usd_path",
    type=str,
    default="/localhome/tea21/Desktop/environment_without_people.usd",
    help="USD to spawn per env",
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
from pxr import UsdGeom, Gf, UsdPhysics, Usd, Sdf  # USD APIs (version-safe)
from omni.physx.scripts import utils as physx_utils  # GPU-safe kinematic motion

# ---------------- USER EDITS ----------------
REL_LIDAR_PATH = "Environment/turtlebot/turtlebot3_burger/Lidar"
REL_BASE_PATH  = "Environment/turtlebot/turtlebot3_burger/base_footprint"
# -------------------------------------------

# Physics settings: keep GPU pipeline ON
settings = carb.settings.get_settings()
settings.set("/physics/use_gpu", True)
settings.set("/physics/cudaDevice", 0)
settings.set("/physics/tensors/device", 0)
settings.set("/physics/use_gpu_pipeline", True)   # Direct GPU API stays enabled

# ---- Movers / arena params ----
NUM_MOVERS_PER_ENV = 4
BOX_SIZE           = 0.25      # m
SPEED_MIN          = 0.6       # m/s
SPEED_MAX          = 1.4       # m/s
JITTER_STD         = 0.25      # rad/s-equivalent heading jitter
BOUNCE_GAIN        = 1.0       # reflection gain
ARENA_HALF         = 5.0       # env-local square half-size in XY
WALL_THICK         = 0.05      # m
WALL_HEIGHT        = 0.5       # m
SEED               = 42

np.random.seed(SEED)

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Loads your USD asset once per env."""
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

def get_world_pose(stage, path: str):
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return None, None
    xf = UsdGeom.Xformable(prim)
    wt = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = wt.ExtractTranslation()
    q = wt.ExtractRotation().GetQuaternion()
    pos = np.array([t[0], t[1], t[2]], dtype=float)
    quat_xyzw = np.array([q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2], q.GetReal()], dtype=float)
    return pos, quat_xyzw

def get_world_vel(stage, path: str):
    rb = UsdPhysics.RigidBodyAPI.Get(stage, path)
    if not rb:
        return None, None
    lin = rb.GetVelocityAttr().Get()
    ang = rb.GetAngularVelocityAttr().Get()
    if lin is None or ang is None:
        return None, None
    lin = np.array([lin[0], lin[1], lin[2]], dtype=float)
    ang = np.array([ang[0], ang[1], ang[2]], dtype=float)
    return lin, ang

# ---- Idempotent USD helpers ----
def _mk_xform(path: Sdf.Path):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        prim = stage.DefinePrim(path, "Xform")
    return UsdGeom.Xformable(prim)

def _spawn_cube(path: str, center_xyz, size_xyz, visible=True):
    """
    Spawn (or update) an Xform + UsdGeom.Cube at `path`, scaled to size_xyz and translated to center_xyz.
    Reuses xform ops if present (avoid duplicate op errors).
    """
    stage = omni.usd.get_context().get_stage()
    _mk_xform(Sdf.Path(path))
    geom_path = f"{path}/geom"
    cube = UsdGeom.Cube.Get(stage, Sdf.Path(geom_path))
    if not cube:
        cube = UsdGeom.Cube.Define(stage, Sdf.Path(geom_path))
        cube.CreateSizeAttr(1.0)  # scale via xform ops

    xformable = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    existing_ops = {op.GetOpName(): op for op in xformable.GetOrderedXformOps()}
    xf_op = existing_ops.get("xformOp:transform") or xformable.AddTransformOp()
    sc_op = existing_ops.get("xformOp:scale") or xformable.AddScaleOp()
    xformable.SetXformOpOrder([xf_op, sc_op])
    xf_op.Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(*center_xyz)))
    sc_op.Set(Gf.Vec3f(size_xyz[0], size_xyz[1], size_xyz[2]))

    cube.CreateVisibilityAttr().Set("inherited" if visible else "invisible")
    return stage.GetPrimAtPath(path)

def _get_or_create_ops(path: str):
    stage = omni.usd.get_context().get_stage()
    xformable = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    ops = {op.GetOpName(): op for op in xformable.GetOrderedXformOps()}
    xf_op = ops.get("xformOp:transform") or xformable.AddTransformOp()
    sc_op = ops.get("xformOp:scale") or xformable.AddScaleOp()
    xformable.SetXformOpOrder([xf_op, sc_op])
    return xf_op, sc_op

# (kept for reference; not used to move actors in GPU mode)
def _set_local_xy(path: str, x: float, y: float, z_fixed: float):
    xf_op, _ = _get_or_create_ops(path)
    xf_op.Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(x, y, z_fixed)))

# GPU-pipeline-safe kinematic motion
def _set_kinematic_target_xy(path: str, x: float, y: float, z_fixed: float):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return
    tf = Gf.Matrix4d().SetTranslate(Gf.Vec3d(x, y, z_fixed))
    try:
        physx_utils.set_kinematic_target(prim, tf)
    except Exception as e:
        # If actor isn't cooked yet, just set xform (will be correct after warm-up)
        # and print once.
        global _warned_kt
        if "_warned_kt" not in globals():
            print(f"[WARN] set_kinematic_target failed on {path}: {e}")
            _warned_kt = True
        _set_local_xy(path, x, y, z_fixed)

# ---- Colliders (geom-level) ----
def _add_static_wall(path, center_xyz, size_xyz):
    """Invisible static wall with collider on GEOM so PhysX raycasts (and LiDAR) can hit it."""
    prim = _spawn_cube(path, center_xyz, size_xyz, visible=False)
    stage = omni.usd.get_context().get_stage()
    geom_prim = stage.GetPrimAtPath(f"{path}/geom")
    UsdPhysics.CollisionAPI.Apply(geom_prim)
    geom_prim.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set("box")
    return prim

def _add_moving_box_kinematic(path, center_xyz, edge_len, color=(0.6, 0.9, 0.9)):
    """
    Kinematic box:
      - RigidBody parent (PhysX actor),
      - Collider on GEOM with 'box' approx (PhysX LiDAR hit),
      - Moved each tick via kinematic target (robot collides with it but can't push it).
    """
    prim = _spawn_cube(path, center_xyz, (edge_len, edge_len, edge_len), visible=True)

    stage = omni.usd.get_context().get_stage()
    geom_path = f"{path}/geom"
    geom_prim = stage.GetPrimAtPath(geom_path)

    # Visual color
    if geom_prim and geom_prim.IsValid():
        UsdGeom.Gprim(geom_prim).CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])

    # Collider on GEOM + explicit box approximation (critical for LiDAR)
    UsdPhysics.CollisionAPI.Apply(geom_prim)
    geom_prim.CreateAttribute("physics:approximation", Sdf.ValueTypeNames.Token).Set("box")

    # Parent is a rigid body; mark it kinematic
    UsdPhysics.RigidBodyAPI.Apply(prim)
    prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(True)

    return prim

def _env_root_path(i: int) -> str:
    return f"/World/envs/env_{i}"

# ---------------- Per-env state ----------------
_env_boxes   = {}  # i -> list of mover prim paths
_env_velxy   = {}  # i -> (N,2) target planar velocities (env-local)
_env_locpos  = {}  # i -> (N,2) local positions (env-local)

def _build_arena_and_boxes_for_env(i: int):
    """Create 4 walls and 4 movers (kinematic RBs) inside env i (env-local coords)."""
    root = _env_root_path(i)
    arena_parent  = f"{root}/Arena"
    movers_parent = f"{root}/Movers"

    L = ARENA_HALF
    t = WALL_THICK
    h = WALL_HEIGHT
    zc = 0.5 * h

    # Walls
    _add_static_wall(f"{arena_parent}/Wall_PosX", [ L,      0.0, zc], [t, 2*L + 2*t, h])
    _add_static_wall(f"{arena_parent}/Wall_NegX", [-L,     0.0, zc], [t, 2*L + 2*t, h])
    _add_static_wall(f"{arena_parent}/Wall_PosY", [ 0.0,   L,   zc], [2*L + 2*t, t, h])
    _add_static_wall(f"{arena_parent}/Wall_NegY", [ 0.0,  -L,   zc], [2*L + 2*t, t, h])

    # Movers
    boxes, velxy, locpos = [], [], []
    margin = BOX_SIZE * 0.5 + 0.2
    z_fixed = BOX_SIZE * 0.5 + 0.01
    for k in range(NUM_MOVERS_PER_ENV):
        x = np.random.uniform(-L + margin, L - margin)
        y = np.random.uniform(-L + margin, L - margin)
        path = f"{movers_parent}/box_{k:02d}"
        color = tuple(np.random.uniform(0.25, 0.9, size=3))
        _add_moving_box_kinematic(path, [x, y, z_fixed], BOX_SIZE, color=color)
        boxes.append(path)
        locpos.append([x, y])

        speed = np.random.uniform(SPEED_MIN, SPEED_MAX)
        theta = np.random.uniform(-np.pi, np.pi)
        velxy.append([np.cos(theta) * speed, np.sin(theta) * speed])

    _env_boxes[i]  = boxes
    _env_velxy[i]  = np.array(velxy, dtype=np.float32)
    _env_locpos[i] = np.array(locpos, dtype=np.float32)

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

    # Spawn per-env arena and movers after a frame so env_xform exists
    omni.kit.app.get_app().update()
    for i in range(args_cli.num_envs):
        _build_arena_and_boxes_for_env(i)

    # >>> Warm-up cook: ensure PhysX actors exist BEFORE we set kinematic targets
    scene.write_data_to_sim()
    sim.step()

    scan_counter = 0
    step_counter = 0

    # Local bounds for reflection
    half = BOX_SIZE * 0.5
    local_min = -(ARENA_HALF - half)
    local_max = +(ARENA_HALF - half)
    z_fixed = BOX_SIZE * 0.5 + 0.01

    while simulation_app.is_running():
        dt = float(sim.get_physics_dt())

        # 1) Animate movers FIRST (so PhysX sees new kinematic targets this step)
        for i in range(args_cli.num_envs):
            if i not in _env_boxes:
                continue
            vel = _env_velxy[i]
            pos = _env_locpos[i]

            # jitter small heading angles (random-walk feel)
            jitter = np.random.randn(len(vel)) * JITTER_STD * dt
            cj, sj = np.cos(jitter), np.sin(jitter)
            vx, vy = vel[:, 0].copy(), vel[:, 1].copy()
            vel[:, 0] = cj * vx - sj * vy
            vel[:, 1] = sj * vx + cj * vy

            # integrate and reflect inside bounds
            pos[:, 0] += vel[:, 0] * dt
            pos[:, 1] += vel[:, 1] * dt

            over_min = pos[:, 0] < local_min
            over_max = pos[:, 0] > local_max
            vel[over_min | over_max, 0] *= -BOUNCE_GAIN
            pos[over_min, 0] = local_min + (local_min - pos[over_min, 0])
            pos[over_max, 0] = local_max - (pos[over_max, 0] - local_max)

            over_min = pos[:, 1] < local_min
            over_max = pos[:, 1] > local_max
            vel[over_min | over_max, 1] *= -BOUNCE_GAIN
            pos[over_min, 1] = local_min + (local_min - pos[over_min, 1])
            pos[over_max, 1] = local_max - (pos[over_max, 1] - local_max)

            # write GPU-safe kinematic targets BEFORE stepping
            for k, prim_path in enumerate(_env_boxes[i]):
                _set_kinematic_target_xy(prim_path, float(pos[k, 0]), float(pos[k, 1]), z_fixed)

        # 2) Push USD->PhysX and step
        scene.write_data_to_sim()
        sim.step()
        scan_counter += 1
        step_counter += 1

        # -------- Optional LiDAR prints ----------
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
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)

if __name__ == "__main__":
    try:
        main()
    finally:
        # Graceful shutdown: stop sim loops, detach stage, then close the app.
        ctx = omni.usd.get_context()
        if ctx and ctx.get_stage():
            ctx.close_stage()
        try:
            omni.kit.app.get_app().update()
        except Exception:
            pass
        simulation_app.close()

