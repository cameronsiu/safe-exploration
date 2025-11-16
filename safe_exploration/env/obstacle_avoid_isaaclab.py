from __future__ import annotations
from typing import TYPE_CHECKING

import gym
from gym.spaces import Box, Dict
import numpy as np
import torch

if TYPE_CHECKING:
    from isaacsim.simulation_app import SimulationApp
    from isaaclab.sim import SimulationContext
    from isaaclab.scene import InteractiveScene
    from isaaclab.assets.articulation import Articulation


from safe_exploration.core.config import Config

class ObstacleAvoidIsaacLab(gym.Env):

    def __init__(self, sim_app: SimulationApp, sim_context: SimulationContext, scene: InteractiveScene):

        self._config = Config.get().env.obstacleavoidisaaclab
        self._action_scale = self._config.action_scale

        # NOTE: use IsaacSim to change the lidar in simulation and replace parameter in yaml
        self.num_lidars = self._config.num_lidars

        # NOTE: turtlebot will apply velocity commands to wheel joints independently
        # TODO: We can change this to be more like ROS2
        turtlebot_max_speed = 6
        self.action_space = Box(low=-turtlebot_max_speed, high=turtlebot_max_speed, shape=(2,), dtype=np.float32)
        
        # NOTE: Not sure to include z value
        self.observation_space = Dict({
            'agent_position': Box(low=-10, high=10, shape=(2,), dtype=np.float32),
            'target_position': Box(low=-8, high=8, shape=(2,), dtype=np.float32),
            'lidar_readings': Box(low=0.2, high=15, shape=(self.num_lidars,), dtype=np.float32)
        })

        ## Isaac Sim
        self.sim_app = sim_app
        self.sim_context = sim_context
        self.sim_dt = self.sim_context.get_physics_dt()
        self.scene = scene

        from isaacsim.sensors.physx import _range_sensor

        self.lidar_interface = _range_sensor.acquire_lidar_sensor_interface()
        # TODO: hardcode for now, not sure how to get prim paths properly here
        self.lidar_prim_path = "/World/envs/env_0/Turtlebot/turtlebot3_burger/base_footprint/base_link/base_scan/Lidar"


    def _get_reward(self, initial_position: np.ndarray, final_position: np.ndarray):
        if self._config.enable_reward_shaping and self._is_agent_outside_shaping_boundary():
            return -1
        else:
            # Square each distance so being closer is more valuable
            sq_initial_distance = (1 - np.linalg.norm(initial_position - self._target_position)) ** 2
            sq_final_distance = (1 - np.linalg.norm(final_position - self._target_position)) ** 2
            distance_change = sq_final_distance - sq_initial_distance

            return distance_change

    def _get_lidar_readings(self) -> np.ndarray:
        depth: np.ndarray = self.lidar_interface.get_linear_depth_data(self.lidar_prim_path).reshape(-1)
        return depth

    def _get_noisy_target_position(self):
        return self._target_position + \
               np.random.normal(0, self._config.target_noise_std, 2)

    def _reset_target_location(self):
        agent_on_top = self._agent_position[1] > 0.0

        # TODO: Fix this
        #target_position = np.random.random(2)

        if agent_on_top:
            self._target_position = np.array([-8.0, -8.0]) #+ target_position * np.array([8.0, -5.0])
        else:
            self._target_position = np.array([-8.0, 8.0]) #+ target_position * np.array([8.0, -2.5])

        # NOTE: Just resetting the scene not the entire simulation context
        self.scene.reset()

    def _did_agent_collide(self) -> bool:
        # TODO: Check if Isaac Sim collision can return true/false
        boundary = (self._agent_position <= -10) | (self._agent_position >= 10)
        return np.any(boundary)
    
    def _is_agent_outside_shaping_boundary(self):
        return np.any(self._get_lidar_readings() < self._config.reward_shaping_slack)

    def _update_time(self):
        # Assume that frequency of motor is 1 (one action per second)
        # NOTE: Not sure if this should be same as self.sim_dt
        self._current_time += self.sim_context.get_physics_dt()

    def get_num_constraints(self):
        return 1

    def get_constraint_values(self):
        clipped_readings = np.clip(self._get_lidar_readings(), 0, 0.2)
        return np.array([self._config.agent_slack - np.min(clipped_readings)])

    def reset(self):
        """
        Ref: https://isaac-sim.github.io/IsaacLab/main/source/tutorials/01_assets/add_new_robot.html
        """
        self.sim_context.reset()
        turtlebot: Articulation = self.scene["Turtlebot"]

        agent_position = np.random.random(2)
        target_position = np.random.random(2)
        agent_on_top = np.random.random(1)[0] > 0

        if agent_on_top:
            self._agent_position = np.array([-8.0, 8.0]) #+ agent_position * np.array([0.8, -0.25])
            self._target_position = np.array([-0.8, -0.8]) #+ target_position * np.array([0.8, 0.25])
        else:
            self._agent_position = np.array([-8.0, -8.0]) #+ agent_position * np.array([0.8, 0.25])
            self._target_position = np.array([-8.0, 8.0]) #+ target_position * np.array([0.8, -0.25])

        root_turtlebot_state = turtlebot.data.default_root_state.clone()
        root_turtlebot_state[:, :3] += self.scene.env_origins
        root_turtlebot_state[:, :2] += self._agent_position

        turtlebot.write_root_pose_to_sim(root_turtlebot_state[:, :7])
        turtlebot.write_root_velocity_to_sim(root_turtlebot_state[:, 7:])

        joint_pos, joint_vel = (
            turtlebot.data.default_joint_pos.clone(),
            turtlebot.data.default_joint_vel.clone(),
        )
        turtlebot.write_joint_state_to_sim(joint_pos, joint_vel)

        self._current_time = 0.

        return self.step(np.zeros(2))[0]

    def step(self, action: np.ndarray):
        if self.sim_app.is_running():

            turtlebot: Articulation = self.scene["Turtlebot"]
            initial_position = turtlebot.data.root_pos_w.reshape(-1)[:2].cpu().numpy()

            if (int(100 * self._current_time) // 10) % (self._config.respawn_interval * 10) == 0:
                self._reset_target_location()

            self._update_time()

            turtlebot.set_joint_velocity_target(torch.tensor(action))
            self.scene.write_data_to_sim()
            self.sim_context.step()

            self._agent_position = turtlebot.data.root_pos_w.reshape(-1)[:2].cpu().numpy()

            reward = self._get_reward(initial_position, self._agent_position)

            lidar_readings: np.ndarray = self._get_lidar_readings()

            observation = {
                "agent_position": self._agent_position,
                "target_position": self._get_noisy_target_position(),
                "lidar_readings": lidar_readings
            }

            # TODO: Check if IsaacLab collider returns true/false
            done = self._did_agent_collide() \
                or int(self._current_time // 1) > self._config.episode_length

            # TODO: Not sure if this should be here
            self.scene.update(self.sim_dt)

            return observation, reward, done, {}

