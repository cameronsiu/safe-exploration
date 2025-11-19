# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

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
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ContactSensor

from isaacsim.sensors.physx import _range_sensor

TURTLEBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=f"safe_exploration/env/turtlebot.usd", activate_contact_sensors=True),
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
    )

    Turtlebot: ArticulationCfg = TURTLEBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Turtlebot")


#HINT: Make sure to enable 'activate_contact_sensors' in the corresponding asset spawn configuration.
    contact_forces_RW = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Turtlebot/turtlebot3_burger/wheel_right_link",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Environment/Walls"],
    )

    contact_forces_LW = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Turtlebot/turtlebot3_burger/wheel_left_link",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Environment/Walls"],
    )

    contact_forces_B = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Turtlebot/turtlebot3_burger/base_footprint",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Environment/Walls"],
    )


def debug_turtlebot(turtlebot: Articulation):
    """Just for debuggint he turtlebot's position and velocity"""
    # TODO: agent position would be turtlebot.data.root_pos_w[0] and turtlebot.data.root_pos_w[1]
    print("Turtlebot root state: ", turtlebot.data.root_pos_w)
    #print("Turtlebot joint vel: ", turtlebot.data.root_vel_w)
    print("Turtlebot linear joint vel: ", turtlebot.data.root_lin_vel_w)
    #print("Turtlebot angular joint vel: ", turtlebot.data.root_ang_vel_w)


def reset_turtlebot(scene: InteractiveScene, turtlebot: Articulation):
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

    turtlebot: Articulation = scene["Turtlebot"]

    reset_turtlebot(scene, turtlebot)
    
    lidar_interface = _range_sensor.acquire_lidar_sensor_interface()

    # TODO: hardcode for now, not sure how to get prim paths properly here
    lidar_prim_path = "/World/envs/env_0/Turtlebot/turtlebot3_burger/base_footprint/base_link/base_scan/Lidar"
    NUM_LIDARS = 300 // 10 # 10 is the amount we skip

    count = 0

    while simulation_app.is_running():
        if args_cli.debug:
            debug_turtlebot(turtlebot)

        if count == 0:
            action = torch.Tensor([[6.0, 6.0]])
            turtlebot.set_joint_velocity_target(action)
            scene.write_data_to_sim()

        # reset
        # if count % 500 == 0:
        #     # reset counters
        #     count = 0
            
        #     # reset turtlebot
        #     reset_turtlebot(scene, turtlebot)
            
        #     # clear internal buffers
        #     scene.reset()
        #     print("[INFO]: Resetting Turtlebot state...")

        # NOTE: cameron - ROS2 uses cmd_vel, so there must a conversion from 
        # joint velocity to linear/angular velocity
        # v = w * r,  r = 0.325 radius for the turtlebot
        # [-6.7, 6.7]
        # [6.9, -6.9]

        # if count % 100 < 75:
        #     # Drive straight by setting equal wheel velocities
        #     action = torch.Tensor([[6.0, 6.0]])
        # else:
        #     # Turn by applying different velocities
        #     action = torch.Tensor([[5.0, -5.0]])

        # NOTE: depth data from IsaacSim is a numpy array
        depth: np.ndarray = lidar_interface.get_linear_depth_data(lidar_prim_path)
        depth = depth.reshape(-1)[::NUM_LIDARS]

        # contact_forces_base: ContactSensor = scene["contact_forces_B"]
        # contact_forces_left_wheel: ContactSensor = scene["contact_forces_LW"]
        # contact_forces_right_wheel: ContactSensor = scene["contact_forces_RW"]

        # collide_with_obstacle = (contact_forces_base.data.force_matrix_w.abs() > 1e-4) | \
        #                         (contact_forces_left_wheel.data.force_matrix_w.abs() > 1e-4) | \
        #                         (contact_forces_right_wheel.data.force_matrix_w.abs() > 1e-4)

        # if torch.any(collide_with_obstacle):
        #     # print information from the sensors
        #     print("-------------------------------")
        #     print(scene["contact_forces_RW"])
        #     print("Received force matrix of: ", scene["contact_forces_RW"].data.force_matrix_w)
        #     print("Received contact force of: ", scene["contact_forces_RW"].data.net_forces_w)
        #     print("-------------------------------")
        #     print(scene["contact_forces_LW"])
        #     print("Received force matrix of: ", scene["contact_forces_LW"].data.force_matrix_w)
        #     print("Received contact force of: ", scene["contact_forces_LW"].data.net_forces_w)
        #     print("-------------------------------")
        #     print(scene["contact_forces_B"])
        #     print("Received force matrix of: ", scene["contact_forces_B"].data.force_matrix_w)
        #     print("Received contact force of: ", scene["contact_forces_B"].data.net_forces_w)
            
        #     print(collide_with_obstacle)
        #     # print(contact_forces_base.data.net_forces_w)
        #     # print(contact_forces_left_wheel.data.net_forces_w)
        #     # print(contact_forces_right_wheel.data.net_forces_w)


        sim.step()
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # Design scene
    scene_cfg = ObstacleAvoidCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()