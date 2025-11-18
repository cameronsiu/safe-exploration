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
    from isaaclab.sensors import ContactSensor

from safe_exploration.core.config import Config

class ObstacleAvoidIsaacLab(gym.Env):

    def __init__(self, sim_app: SimulationApp, sim_context: SimulationContext, scene: InteractiveScene, render_step: bool):
        self._config = Config.get().env.obstacleavoidisaaclab
        self._action_scale = self._config.action_scale

        # NOTE: use IsaacSim to change the lidar in simulation and replace parameter in yaml
        self._num_lidar_buckets = self._config.num_lidars

        # NOTE: turtlebot will apply velocity commands to wheel joints independently
        # TODO: We can change this to be more like ROS2
        self.action_space = Box(low=-self._action_scale, high=self._action_scale, shape=(2,), dtype=np.float32)
        
        # NOTE: Not sure to include z value
        # TODO: Use parameters for boundaries
        self.observation_space = Dict({
            'agent_position': Box(low=-10, high=10, shape=(2,), dtype=np.float32),
            'target_position': Box(low=-8, high=8, shape=(2,), dtype=np.float32),
            'lidar_readings': Box(low=0.2, high=15, shape=(self._num_lidar_buckets,), dtype=np.float32)
        })

        ## Isaac Sim
        self.sim_app = sim_app
        self.sim_context = sim_context
        self.sim_dt = self._config.sim_dt
        self.scene = scene

        # Render Sim steps
        self.render_step = render_step

        from isaacsim.sensors.physx import _range_sensor
        from safe_exploration.env.utils.obstacles_utils import _build_boxes_for_env, move_obstacles

        self.lidar_interface = _range_sensor.acquire_lidar_sensor_interface()
        # TODO: hardcode for now, not sure how to get prim paths properly here
        self.lidar_prim_path = "/World/envs/env_0/Turtlebot/turtlebot3_burger/base_footprint/base_link/base_scan/Lidar"

        self.boxes_dict = _build_boxes_for_env()
        self.move_obstacles = move_obstacles

        self.reset()

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
        if self._lidar_readings is not None and self._current_time == self._lidar_measure_time:
            return self._lidar_readings
        elif self._did_agent_collide():
            self._lidar_readings = np.zeros((self._num_lidar_buckets,))
            self._lidar_measure_time = self._current_time
            return self._lidar_readings
        else:
            self._lidar_readings: np.ndarray = self.lidar_interface.get_linear_depth_data(self.lidar_prim_path).reshape(-1)
            self._lidar_measure_time = self._current_time
            return self._lidar_readings

    def _get_noisy_target_position(self):
        return self._target_position + \
               np.random.normal(0, self._config.target_noise_std, 2)

    def _reset_target_location(self):
        agent_y = self._agent_position[1]

        # If agent is in top half, target goes bottom
        if agent_y > 0:
            self._target_position = self._sample_position(-10, -2)
        else:
            self._target_position = self._sample_position(2, 10)

    def _did_agent_collide(self) -> bool:
        contact_forces_base: ContactSensor = self.scene["contact_forces_B"]
        contact_forces_left_wheel: ContactSensor = self.scene["contact_forces_LW"]
        contact_forces_right_wheel: ContactSensor = self.scene["contact_forces_RW"]

        collide_with_obstacle = (contact_forces_base.data.force_matrix_w != 0.0) | \
                                (contact_forces_left_wheel.data.force_matrix_w != 0.0) | \
                                (contact_forces_right_wheel.data.force_matrix_w != 0.0)

        return torch.any(collide_with_obstacle)
    
    def _is_agent_outside_shaping_boundary(self):
        return np.any(self._get_lidar_readings() < self._config.reward_shaping_slack)

    def _update_time(self):
        # Assume that frequency of motor is 1 (one action per second)
        # TODO: Not sure if this is correct
        self._current_time += self._config.sim_dt

    def _sample_position(self, y_min: float, y_max: float, margin:float=1.0):
        """
        Sample a (x,y) in [-10,10] x [-10,10] but restricted to a y-band.
        margin avoids spawning at walls.
        """
        x = np.random.uniform(-10 + margin, 10 - margin)
        y = np.random.uniform(y_min + margin, y_max - margin)
        return np.array([x, y], dtype=np.float32)

    def get_num_constraints(self):
        return 1

    def get_constraint_values(self):
        clipped_readings = np.clip(self._get_lidar_readings(), 0, 0.2)
        return np.array([self._config.agent_slack - np.min(clipped_readings)])

    def reset(self):
        """
        Resets the agent in either the top or bottom half of the map,
        and places the target in the opposite half.
        Coordinate system:
            x ∈ [-10,10]
            y ∈ [-10,10]   (origin at center)
        """
        turtlebot: Articulation = self.scene["Turtlebot"]

        # Randomly decide agent region (top or bottom)
        agent_on_top = np.random.rand() > 0.5

        if agent_on_top:
            self._agent_position = self._sample_position(2, 10)
            self._target_position = self._sample_position(-10, -2)
        else:
            self._agent_position = self._sample_position(-10, -2)
            self._target_position = self._sample_position(2, 10)

        # Write the agent pose into IsaacSim
        root_state = turtlebot.data.default_root_state.clone()
        root_state[:, :3] += self.scene.env_origins
        agent_position = torch.tensor(self._agent_position)
        root_state[:, 0] = agent_position[0]
        root_state[:, 1] = agent_position[1]

        turtlebot.write_root_pose_to_sim(root_state[:, :7])
        turtlebot.write_root_velocity_to_sim(root_state[:, 7:])

        # Reset joints
        joint_pos = turtlebot.data.default_joint_pos.clone()
        joint_vel = turtlebot.data.default_joint_vel.clone()
        turtlebot.write_joint_state_to_sim(joint_pos, joint_vel)

        # Reset timers
        self._current_time = 0.0
        self._lidar_readings = None
        self._lidar_measure_time = -1.0

        # Reset only environment physics (not entire sim)
        self.scene.reset()

        # Return initial observation
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
            self.sim_context.step(self.render_step)
            self.move_obstacles(self.sim_dt, self.boxes_dict)

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
        
    def render_env(self):
        """Isaac Sim is already rendered"""
        pass

