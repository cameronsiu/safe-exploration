import numpy as np
import omni

from pxr import UsdGeom, Gf, UsdPhysics, Usd, Sdf
from omni.physx.scripts import utils as physx_utils

BOX_SIZE           = 1.0       # m
SPEED_MIN          = 0.2       # m/s
SPEED_MAX          = 0.2       # m/s
JITTER_STD         = 0.25      # rad/s-equivalent heading jitter
BOUNCE_GAIN        = 1.0       # reflection gain
ARENA_HALF         = 10.0      # env-local square half-size in XY
SEED               = 42

np.random.seed(SEED)

## Creating Obstacles
def build_boxes_for_env(num_obstacles: int, obstacles_prim_path: str):
    """Create 4 walls and 4 moving obstacles (kinematic RBs) inside env (env-local coords)."""
    L = ARENA_HALF

    # Movers
    boxes, velxy, locpos = [], [], []
    margin = BOX_SIZE * 0.5 + 0.2
    z_fixed = BOX_SIZE * 0.5 + 0.01
    for k in range(num_obstacles):
        x = np.random.uniform(-L + margin, L - margin)
        y = np.random.uniform(-L + margin, L - margin)
        path = f"{obstacles_prim_path}/box_{k}"
        color = tuple(np.random.uniform(0.25, 0.9, size=3))
        boxes.append(path)
        locpos.append([x, y])

        speed = np.random.uniform(SPEED_MIN, SPEED_MAX)
        theta = np.random.uniform(-np.pi, np.pi)
        velxy.append([np.cos(theta) * speed, np.sin(theta) * speed])

    boxes_dict = {
        "box_paths": boxes,
        "velxy": np.array(velxy, dtype=np.float32),
        "locpos": np.array(locpos, dtype=np.float32)
    }

    return boxes_dict

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

def move_obstacles(sim_dt: float, boxes_dict: dict):
    # Local bounds for reflection
    half = BOX_SIZE * 0.5
    local_min = -(ARENA_HALF - half)
    local_max = +(ARENA_HALF - half)
    z_fixed = BOX_SIZE * 0.5 + 0.01

    dt = sim_dt

    # 1) Animate movers FIRST (so PhysX sees new kinematic targets this step)
    vel = boxes_dict["velxy"]
    pos = boxes_dict["locpos"]

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

    for k, prim_path in enumerate(boxes_dict["box_paths"]):
        _set_kinematic_target_xy(prim_path, float(pos[k, 0]), float(pos[k, 1]), z_fixed)
