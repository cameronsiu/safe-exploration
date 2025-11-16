import argparse

from isaaclab.app import AppLauncher
from isaacsim.simulation_app import SimulationApp

parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--debug", type=bool, default=False, help="Debug turtlebot pos and vel.")
parser.add_argument("--device", type=str, default='cpu', help="Device.")
parser.add_argument("--headless", type=bool, default=True, help="Device.")

args_cli = parser.parse_args()
print(args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app: SimulationApp = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

simulation_app.update()

from isaacsim.sensors.physx import _range_sensor

TURTLEBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"safe_exploration/env/turtlebot.usd"),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)

class ObstacleAvoidCfg(InteractiveSceneCfg):
    """Obstacle Avoid Scene."""

    scene: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Environment",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)
        ),
        spawn=sim_utils.UsdFileCfg(usd_path="safe_exploration/env/obstacle_avoid.usd"),
        debug_vis=True,
    )

    Turtlebot: ArticulationCfg = TURTLEBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Turtlebot")


def debug_turtlebot(turtlebot: Articulation, depth: np.ndarray):
    """Just for debuggint he turtlebot's position and velocity"""
    # TODO: agent position would be turtlebot.data.root_pos_w[0] and turtlebot.data.root_pos_w[1]
    # shape [1, 3]
    x, y = turtlebot.data.root_pos_w[:, 0], turtlebot.data.root_pos_w[:, 1]
    print(f"Turtlebot xy: {x} {y}")
    #print("Turtlebot joint vel: ", turtlebot.data.root_vel_w)

    # shape [1, 3]
    x_vel, y_vel = turtlebot.data.root_lin_vel_w[:, 0], turtlebot.data.root_lin_vel_w[:, 1]
    print(f"Turtlebot linear joint vel: {x_vel} {y_vel}")
    #print("Turtlebot angular joint vel: ", turtlebot.data.root_ang_vel_w)

    # do not turn on draw lidar
    if len(depth) > 0:
        print(f"[LiDAR] scan rays={len(depth)} | shape={depth.shape} "
              f"min={float(np.min(depth)):.3f} m | max={float(np.max(depth)):.3f} m")

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    count = 0
    
    turtlebot: Articulation = scene["Turtlebot"]

    # Play the simulator
    sim.reset()
    
    lidar_interface = _range_sensor.acquire_lidar_sensor_interface()

    # TODO: hardcode for now, not sure how to get prim paths properly here
    lidar_prim_path = "/World/envs/env_0/Turtlebot/turtlebot3_burger/base_footprint/base_link/base_scan/Lidar"

    while simulation_app.is_running():
        # NOTE: depth data from IsaacSim is a numpy array
        # Can we return a torch tensor?
        depth: np.ndarray = lidar_interface.get_linear_depth_data(lidar_prim_path)

        debug_turtlebot(scene["Turtlebot"], depth)

        if count % 500 == 0:
            count = 0
            root_turtlebot_state = turtlebot.data.default_root_state.clone()
            root_turtlebot_state[:, :3] += scene.env_origins

            turtlebot.write_root_pose_to_sim(root_turtlebot_state[:, :7])
            turtlebot.write_root_velocity_to_sim(root_turtlebot_state[:, 7:])

            joint_pos, joint_vel = (
                turtlebot.data.default_joint_pos.clone(),
                turtlebot.data.default_joint_vel.clone(),
            )
            turtlebot.write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()

        # NOTE: cameron - ROS2 uses cmd_vel, so there must a conversion from 
        # joint velocity to linear/angular velocity
        # v = w * r,  r = 0.325 radius for the turtlebot
        # [-6.7, 6.7]
        # [6.9, -6.9]

        #turtlebot.set_joint_velocity_target(action)
        #scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

def main():
    """Main function."""
    # Initialize the simulation context
    # render_cfg = sim_utils.RenderCfg(rendering_mode="quality")
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, enable_scene_query_support=True)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    # Design scene
    scene_cfg = ObstacleAvoidCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()