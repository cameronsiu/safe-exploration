# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# /workspace/isaaclab/apps/isaaclab.python.kit
import argparse

from isaacsim import SimulationApp
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--debug", type=bool, default=False, help="Debug turtlebot pos and vel.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
#app_launcher = AppLauncher(args_cli)
#simulation_app = app_launcher.app

simulation_app = SimulationApp({"headless": False})

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
import isaaclab.assets.articulation as IL_assets
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaacsim.core.api import World


from isaacsim.sensors.physx import _range_sensor
from isaacsim.core.utils.prims import get_prim_at_path, get_prim_children
import isaacsim.core.prims as IS_core_prims
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.viewports import set_camera_view

TURTLEBOT_CONFIG = IL_assets.ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"scripts/tutorials/06_test/turtlebot.usd"),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)

class ObstacleAvoidCfg(InteractiveSceneCfg):
    """Obstacle Avoid Scene."""

    scene: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Environment",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)
        ),
        spawn=sim_utils.UsdFileCfg(usd_path="scripts/tutorials/06_test/obstacle_avoid.usd"),
        debug_vis=True,
    )

    Turtlebot: IL_assets.ArticulationCfg = TURTLEBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Turtlebot")
    #Turtlebot2: ArticulationCfg = TURTLEBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Turtlebot2")


def debug_turtlebot(turtlebot: IL_assets.Articulation):
    """Just for debuggint he turtlebot's position and velocity"""
    # TODO: agent position would be turtlebot.data.root_pos_w[0] and turtlebot.data.root_pos_w[1]
    print("Turtlebot root state: ", turtlebot.data.root_pos_w)
    #print("Turtlebot joint vel: ", turtlebot.data.root_vel_w)
    print("Turtlebot linear joint vel: ", turtlebot.data.root_lin_vel_w)
    #print("Turtlebot angular joint vel: ", turtlebot.data.root_ang_vel_w)


def reset_turtlebot(scene: InteractiveScene, turtlebot: IL_assets.Articulation):
    # reset the scene entities to their initial positions offset by the environment origins
    root_turtlebot_state = turtlebot.data.default_root_state.clone()
    root_turtlebot_state[:, :3] += scene.env_origins

    # copy the default root state to the sim for the turtlebot's orientation and velocity
    turtlebot.write_root_pose_to_sim(root_turtlebot_state[:, :7])
    turtlebot.write_root_velocity_to_sim(root_turtlebot_state[:, 7:])

    # copy the default joint states to the sim
    joint_pos, joint_vel = (
        turtlebot.data.default_joint_pos.clone(),
        turtlebot.data.default_joint_vel.clone(),
    )
    turtlebot.write_joint_state_to_sim(joint_pos, joint_vel)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    count = 0

    # Hardcode path for now
    # turtlebot_usd = "scripts/tutorials/06_test/turtlebot.usd"
    # turtlebot_path = "/World/Turtlebot"
    # add_reference_to_stage(usd_path=turtlebot_usd, prim_path=turtlebot_path)
    # turtlebot = IS_core_prims.Articulation(prim_paths_expr=turtlebot_path, name="turtlebot")
    # turtlebot.set_world_poses(positions=torch.Tensor([[0.0, -1.0, 0.0]]) / get_stage_units())
    
    turtlebot: IL_assets.Articulation = scene["Turtlebot"]

    # Play the simulator
    sim.reset()
    
    lidar_interface = _range_sensor.acquire_lidar_sensor_interface()

    # TODO: hardcode for now, not sure how to get prim paths properly here
    lidar_prim_path = "/World/envs/env_0/Turtlebot/turtlebot3_burger/base_footprint/Lidar"
    NUM_LIDARS = 300 // 10 # 10 is the amount we skip

    while simulation_app.is_running():
        if args_cli.debug:
            debug_turtlebot(scene["Turtlebot"])

        # reset
        if count % 500 == 0:
            # reset counters
            count = 0
            
            # reset turtlebot
            reset_turtlebot(scene, turtlebot)
            
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting Turtlebot state...")

        # NOTE: cameron - ROS2 uses cmd_vel, so there must a conversion from 
        # joint velocity to linear/angular velocity
        # v = w * r,  r = 0.325 radius for the turtlebot
        # [-6.7, 6.7]
        # [6.9, -6.9]

        if count % 100 < 75:
            # Drive straight by setting equal wheel velocities
            action = torch.Tensor([[6.0, 6.0]])
        else:
            # Turn by applying different velocities
            action = torch.Tensor([[5.0, -5.0]])

        # NOTE: depth data from IsaacSim is a numpy array
        depth: np.ndarray = lidar_interface.get_linear_depth_data(lidar_prim_path)
        depth = depth.reshape(-1)[::NUM_LIDARS]

        #turtlebot.set_joint_velocities(action)

        turtlebot.set_joint_velocity_target(action)
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

def main():
    """Main function."""
    # Initialize the simulation context
    # render_cfg = sim_utils.RenderCfg(rendering_mode="quality")
    # sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, render=render_cfg)
    # sim = sim_utils.SimulationContext(sim_cfg)
    #sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    # preparing the scene
    sim = World(stage_units_in_meters=1.0, device=args_cli.device)
    set_camera_view(
        eye=[5.0, 0.0, 1.5], target=[0.00, 0.00, 1.00], camera_prim_path="/OmniverseKit_Persp"
    )  # set camera view

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