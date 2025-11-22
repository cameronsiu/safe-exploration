import numpy as np
import omni

from pxr import UsdGeom, Gf
from omni.physx.scripts import utils as physx_utils

SPEED_MIN          = 0.2       # m/s
SPEED_MAX          = 0.2       # m/s
JITTER_STD         = 0.25      # rad/s-equivalent heading jitter
BOUNCE_GAIN        = 1.0       # reflection gain
SEED               = 42

np.random.seed(SEED)

## Creating Obstacles
def build_obstacles_for_env(num_obstacles: int, obstacles_prim_path: str, obstacle_positions: list):
    obstacles_paths, velxy = [], []
    for k in range(num_obstacles):
        path = f"{obstacles_prim_path}/box_{k}"
        obstacles_paths.append(path)
        speed = np.random.uniform(SPEED_MIN, SPEED_MAX)
        theta = np.random.uniform(-np.pi, np.pi)
        velxy.append([np.cos(theta) * speed, np.sin(theta) * speed])

    obstacles = {
        "obstacles_paths": obstacles_paths,
        "velxy": np.array(velxy, dtype=np.float32),
        "locpos": np.array(obstacle_positions, dtype=np.float32)
    }

    return obstacles

## Changing XForm of obstacles
def _get_or_create_ops(path: str):
    stage = omni.usd.get_context().get_stage()
    xformable = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    ops = {op.GetOpName(): op for op in xformable.GetOrderedXformOps()}
    xf_op = ops.get("xformOp:transform") or xformable.AddTransformOp()
    sc_op = ops.get("xformOp:scale") or xformable.AddScaleOp()
    xformable.SetXformOpOrder([xf_op, sc_op])
    return xf_op, sc_op

def _set_local_xy(path: str, x: float, y: float, z_fixed: float):
    xf_op, _ = _get_or_create_ops(path)
    xf_op.Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(x, y, z_fixed)))

def _set_kinematic_target_xy(path: str, x: float, y: float, z: float):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return
    tf = Gf.Matrix4d().SetTranslate(Gf.Vec3d(x, y, z))
    try:
        physx_utils.set_kinematic_target(prim, tf)
    except Exception as e:
        # If actor isn't cooked yet, just set xform (will be correct after warm-up)
        # and print once.
        global _warned_kt
        if "_warned_kt" not in globals():
            print(f"[WARN] set_kinematic_target failed on {path}: {e}")
            _warned_kt = True
        _set_local_xy(path, x, y, z)

def move_obstacles(sim_dt: float, obstacles: dict, arena_size: int, box_size: int):
    # Local bounds for reflection
    half = box_size * 0.5
    local_min = -(arena_size - half)
    local_max = +(arena_size - half)

    dt = sim_dt

    # 1) Animate movers FIRST (so PhysX sees new kinematic targets this step)
    vel = obstacles["velxy"]
    pos = obstacles["locpos"]

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

    for k, prim_path in enumerate(obstacles["obstacles_paths"]):
        _set_kinematic_target_xy(prim_path, float(pos[k, 0]), float(pos[k, 1]), float(pos[k, 2]))
