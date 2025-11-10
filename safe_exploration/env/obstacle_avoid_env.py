# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Interactive scene with PhysX LiDAR + GT pose/velocity (multi-env).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--usd_path", type=str, default="/localhome/tea21/Desktop/environment_without_people.usd", help="USD to spawn per env")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------- Imports ----------------
import numpy as np
import torch
import carb
import omni
from collections.abc import Sequence

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import DirectRLEnv

from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


from isaaclab.assets import Articulation

from isaacsim.sensors.physx import _range_sensor
from pxr import UsdGeom, Gf, UsdPhysics, Usd

# cameron: please don't use relative imports
from .obstacle_avoid_env_cfg import ObstacleAvoidEnvCfg


REL_LIDAR_PATH = "env.*/turtlebot/turtlebot3_burger/Lidar"            
REL_BASE_PATH  = "env.*/turtlebot/turtlebot3_burger/base_footprint"

my_setting = carb.settings.get_settings()
my_setting.set("/physics/use_gpu", True)
my_setting.set("/physics/cudaDevice", 0)
my_setting.set("/physics/tensors/device", 0)
my_setting.set("/physics/use_gpu_pipeline", True)

# https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_base_env.html
# @configclass
# class ActionsCFG:
#     velocity_commands = mdp.

@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):

        agent_position = ObsTerm()
        lidar_values = ObsTerm()
        target_position = ObsTerm()

        # cameron What do these do?
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # Why structured like this?
    policy: PolicyCfg = PolicyCfg()

# @configclass
# class EventCfg:
#     reset_turtlebot_position = EventTerm(
#         mode="reset",
#         params={
#         }
#     )


# NOTE: What is the difference between DirectRLEnvCFG and ManagerBasedEnvCFG?
@configclass
class ObstacleAvoidEnv(DirectRLEnv):
    cfg: ObstacleAvoidEnvCfg

    def __init__(self, cfg: ObstacleAvoidEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)

    def _setup_scene(self):
        ##### TODO: try device=cuda after .cuda()

        self.robot = Articulation(self.cfg.robot_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.articulations["robot"] = self.robot

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.visualization_markers = define_markers()

        self.up_dir = torch.tensor([0.0, 0.0, 1.0]).cuda()
        self.yaws = torch.zeros((self.cfg.scene.num_envs, 1)).cuda()
        self.commands = torch.randn((self.cfg.scene.num_envs, 3)).cuda()
        self.commands[:, -1] = 0.0
        self.commands = self.commands/torch.linalg.norm(self.commands, dim=1, keepdim=True)

        ratio = self.commands[:, 1]/(self.commands[:,0]+1E-8)
        gzero = torch.where(self.commands > 0, True, False)
        lzero = torch.where(self.commands < 0, True, False)
        plus = lzero[:, 0]*gzero[:, 1]
        minus = lzero[:,0]*lzero[:,1]
        offsets = torch.pi*plus - torch.pi*minus
        self.yaws = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)

        self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset[:,-1] = 0.5
        self.forward_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()
        self.command_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4)).cuda()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # actions shape [num_envs, num_actions]
        self.actions = actions.clone()
        self._visualize_markers()

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        # self.robot is an Articulation
        # self.robot.data is an ArticulationData
        # ArticulationData contains all cloned robots
        self.velocity = self.robot.data.root_com_vel_w
        self.forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)
        obs = torch.hstack((self.velocity, self.commands))
        observations = {"policy": self.velocity}
        return observations
    
    def _get_rewards(self) -> torch.Tensor:
        forward_reward = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
        alignment_reward = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        total_reward = forward_reward + alignment_reward
        return total_reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return False, time_out
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # pick new commands for reset envs
        self.commands[env_ids] = torch.randn((len(env_ids), 3)).cuda()
        self.commands[env_ids,-1] = 0.0
        self.commands[env_ids] = self.commands[env_ids]/torch.linalg.norm(self.commands[env_ids], dim=1, keepdim=True)

        # recalculate the orientations for the command markers with the new commands
        ratio = self.commands[env_ids][:,1]/(self.commands[env_ids][:,0]+1E-8)
        gzero = torch.where(self.commands[env_ids] > 0, True, False)
        lzero = torch.where(self.commands[env_ids]< 0, True, False)
        plus = lzero[:,0]*gzero[:,1]
        minus = lzero[:,0]*lzero[:,1]
        offsets = torch.pi*plus - torch.pi*minus
        self.yaws[env_ids] = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)
    
        # set the root state for the reset envs
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_state_to_sim(default_root_state, env_ids)
        self._visualize_markers()

    def _visualize_markers(self):
        # get marker locations and orientations
        self.marker_locations = self.robot.data.root_pos_w
        self.forward_marker_orientations = self.robot.data.root_quat_w
        self.command_marker_orientations = math_utils.quat_from_angle_axis(self.yaws, self.up_dir).squeeze()

        # offset markers so they are above the jetbot
        loc = self.marker_locations + self.marker_offset
        loc = torch.vstack((loc, loc))
        rots = torch.vstack((self.forward_marker_orientations, self.command_marker_orientations))

        # render the markers
        all_envs = torch.arange(self.cfg.scene.num_envs)
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))
        self.visualization_markers.visualize(loc, rots, marker_indices=indices)


def define_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "forward": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
            "command": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.25, 0.25, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


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
                        #print(f"[LiDAR] scan {scan_counter:05d} | {p} | no data")
                        pass
                    else:
                        # print(f"[LiDAR] scan {scan_counter:05d} | {p} | rays={len(ranges)} | "
                        #       f"min={float(np.min(ranges)):.3f} m | max={float(np.max(ranges)):.3f} m")
                        #print(ranges.shape)  # uncomment to dump full array
                        pass
                except Exception as e:
                    print(f"[LiDAR] scan {scan_counter:05d} | {p} | error: {e}")

            # Ground-truth pose & velocity
            for p in base_paths:
                if not prim_exists(p):
                    continue
                pos, quat = get_world_pose(stage, p)
                lin, ang  = get_world_vel(stage, p)
                if pos is None or lin is None:
                    # print(f"[GT]    scan {scan_counter:05d} | {p} | no data (check that prim is a RigidBody)")
                    pass
                else:
                    # print(
                    #     f"[GT]    scan {scan_counter:05d} | {p} | "
                    #     f"pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) m | "
                    #     f"lin_vel=({lin[0]:+.3f},{lin[1]:+.3f},{lin[2]:+.3f}) m/s | "
                    #     f"ang_vel=({ang[0]:+.3f},{ang[1]:+.3f},{ang[2]:+.3f}) rad/s | "
                    #     f"quat=({quat[0]:+.3f},{quat[1]:+.3f},{quat[2]:+.3f},{quat[3]:+.3f})"
                    # )
                    pass

        # Optional: periodic scene reset
        if step_counter % 500 == 0:
            step_counter = 0
            scene.reset()
            print("[INFO]: Resetting scene...")

def main():
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0")
    sim = SimulationContext(sim_cfg)

    # Camera
    sim.set_camera_view((2.5, 0.0, 4.0), (0.0, 0.0, 2.0))

    # Build scene with multiple envs; spacing so envs don't collide
    scene_cfg = ObstacleAvoidCfg(num_envs=args_cli.num_envs, env_spacing=40.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")

    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()

