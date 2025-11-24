from __future__ import annotations
from typing import TYPE_CHECKING

import gym
from gym.spaces import Box, Dict
import numpy as np
import torch
import time

if TYPE_CHECKING:
    from isaacsim.simulation_app import SimulationApp
    from isaaclab.sim import SimulationContext
    from isaaclab.scene import InteractiveScene
    from isaaclab.assets import RigidObject
    from isaaclab.assets.rigid_object_collection import RigidObjectCollection
    from isaaclab.assets.articulation import Articulation
    from isaaclab.sensors import ContactSensor

from safe_exploration.core.config import Config

class ObstacleAvoidIsaacLab(gym.Env):

    def __init__(self, sim_app: SimulationApp, sim_context: SimulationContext, scene: InteractiveScene):
        self._config = Config.get().env.obstacleavoidisaaclab
        self._action_scale = self._config.action_scale

        # NOTE: use IsaacSim to change the number of lidars and change the parameter in yaml
        self._num_lidar_buckets = self._config.num_lidar_buckets

        # NOTE: turtlebot will apply velocity commands to wheel joints independently
        self.action_space = Box(low=-self._action_scale, high=self._action_scale, shape=(2,), dtype=np.float32)

        self.arena_half = self._config.arena_size // 2
        self.arena_buffer_size = self.arena_half*0.2

        self.observation_space = Dict({
            'agent_position': Box(low=-self.arena_half, high=self.arena_half, shape=(2,), dtype=np.float32),
            'agent_orientation': Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            'agent_velocity': Box(low=-self._action_scale, high=self._action_scale, shape=(2,), dtype=np.float32),
            'target_position': Box(
                low=-(self.arena_half - self.arena_buffer_size),
                high=self.arena_half - self.arena_buffer_size,
                shape=(2,), 
                dtype=np.float32
            ),
            'lidar_readings': Box(low=0.1, high=100, shape=(self._num_lidar_buckets,), dtype=np.float32)
        })

        ## Isaac Sim
        self.sim_app = sim_app
        self.sim_context = sim_context
        self.sim_dt = self.sim_context.get_physics_dt()
        self.scene = scene
        self.action_ratio = self._config.action_ratio

        from isaacsim.sensors.physx import _range_sensor
        from safe_exploration.env.utils.obstacles_utils import build_obstacles_for_env, move_obstacles

        self.lidar_interface = _range_sensor.acquire_lidar_sensor_interface()
        # TODO: hardcode for now, not sure how to get prim paths properly here
        self.lidar_prim_path = "/World/envs/env_0/Turtlebot/turtlebot3_burger/base_footprint/base_link/base_scan/Lidar"

        # NOTE: Hardcoded for now
        self.obstacles_prim_path = f"/World/envs/env_0/Obstacles"
        if self._config.num_obstacles:
            self.obstacle_size = self._config.obstacle_size
            self.obstacles = build_obstacles_for_env(self._config.num_obstacles, self.obstacles_prim_path, self.arena_half, self.obstacle_size)
            self.build_obstacles_for_env = build_obstacles_for_env
            self.move_obstacles = move_obstacles

        self.reset()

    def _get_reward(self, initial_position: np.ndarray, final_position: np.ndarray):
        if self._config.enable_reward_shaping and self._is_agent_outside_shaping_boundary():
            return -1
        else:
            # Square each distance so being closer is more valuable
            initial_distance = np.linalg.norm(initial_position - self._target_position)
            final_distance = np.linalg.norm(final_position - self._target_position)
            neg_distance_change = initial_distance - final_distance

            return neg_distance_change * 10

    def _get_lidar_readings(self) -> np.ndarray:
        if self._lidar_readings is not None and self._current_time == self._lidar_measure_time:
            return self._lidar_readings
        elif self._did_agent_collide():
            self._lidar_readings = np.zeros((self._num_lidar_buckets,))
            self._lidar_measure_time = self._current_time
            return self._lidar_readings
        else:
            raw_lidar_readings: np.ndarray = self.lidar_interface.get_linear_depth_data(self.lidar_prim_path).reshape(-1)
            bucket_size = raw_lidar_readings.shape[0] // self._num_lidar_buckets
            bucketed_lidar_readings = raw_lidar_readings.reshape((self._num_lidar_buckets, bucket_size))
            self._lidar_readings = np.min(bucketed_lidar_readings, axis=1)
            self._lidar_measure_time = self._current_time
            #print(f"minimum lidar values: ", self._lidar_readings, self._agent_position)
            return self._lidar_readings

    def _get_noisy_target_position(self):
        return self._target_position + \
               np.random.normal(0, self._config.target_noise_std, 2)

    def _reset_target_location(self):
        agent_y = self._agent_position[1]

        # If agent is in top half, target goes bottom
        # if agent_y > 0:
        #     self._target_position = self._sample_position(y_min=-1, y_max=0, x_min=-1, x_max=1, margin=0)
        # else:
        #     self._target_position = self._sample_position(y_min=0, y_max=1, x_min=-1, x_max=1, margin=0)
        if agent_y > 0:
            self._target_position = self._sample_position(-self.arena_half, -self.arena_buffer_size, margin=0.2)
        else:
            self._target_position = self._sample_position(self.arena_buffer_size, self.arena_half, margin=0.2)

        target: RigidObject = self.scene["target"]
        target_pos = target.data.default_root_state.clone()
        target_pos[:, :2] = torch.tensor(self._target_position)
        target.write_root_pose_to_sim(target_pos[:, :7])

        # print(self._target_position)

    def _did_agent_collide(self) -> bool:
        contact_forces_base: ContactSensor = self.scene["contact_forces_B"]
        contact_forces_left_wheel: ContactSensor = self.scene["contact_forces_LW"]
        contact_forces_right_wheel: ContactSensor = self.scene["contact_forces_RW"]

        collide_with_obstacle = (contact_forces_base.data.force_matrix_w != 0.0) | \
                                (contact_forces_left_wheel.data.force_matrix_w != 0.0) | \
                                (contact_forces_right_wheel.data.force_matrix_w != 0.0)

        #collide_with_obstacle = self.lidar_interface.get_linear_depth_data(self.lidar_prim_path).reshape(-1) < 0.12

        return torch.any(collide_with_obstacle)
    
    def _is_agent_outside_shaping_boundary(self):
        return np.any(self._get_lidar_readings() < self._config.reward_shaping_slack)

    def _update_time(self):
        self._current_time += self.sim_dt

    def _sample_position(self, y_min: float, y_max: float, margin:float=1.0, x_min: float = None, x_max: float = None):
        """
        Sample a (x,y) in [-self.arena_size,self.arena_size] x [-self.arena_size,self.arena_size]
        but restricted to a y-band. margin avoids spawning at walls.
        """
        x_min = x_min or -self.arena_half
        x_max = x_max or self.arena_half
        x = np.random.uniform(x_min + margin, x_max - margin)
        y = np.random.uniform(y_min + margin, y_max - margin)
        return np.array([x, y], dtype=np.float32)

    def _is_inside_obstacle(self, pos: np.ndarray, safety_margin: float=0.15) -> bool:
        """Check whether a 2D position lies inside any obstacle.
        safety_margin: extra distance buffer around obstacles.
        """
        if not self._config.num_obstacles:
            return False

        obs_xy = self.obstacles["locpos"][:, :2]          # shape (N,2)
        obs_size = self._config.obstacle_size + safety_margin

        # distance from agent to each obstacle center
        dists = np.linalg.norm(obs_xy - pos, axis=1)

        return np.any(dists < obs_size)

    def get_num_constraints(self):
        return 1

    def get_constraint_values(self):
        clipped_readings = np.clip(self._get_lidar_readings(), 0, 0.2)
        return np.array([self._config.agent_slack - np.min(clipped_readings)])

    def reset_turtlebot(self):
        # Write the agent pose into IsaacSim
        turtlebot: Articulation = self.scene["Turtlebot"]
        root_state = turtlebot.data.default_root_state.clone()
        root_state[:, :3] += self.scene.env_origins

        agent_position = torch.tensor(self._agent_position)
        root_state[:, 0:2] = agent_position

        heading = np.random.random() * 2 * np.pi
        q = torch.tensor([np.cos(heading / 2), 0, 0, np.sin(heading / 2)])
        root_state[:, 3:7] = q

        turtlebot.write_root_pose_to_sim(root_state[:, :7])
        turtlebot.write_root_velocity_to_sim(root_state[:, 7:])

        # Reset joints
        joint_pos = turtlebot.data.default_joint_pos.clone()
        joint_vel = turtlebot.data.default_joint_vel.clone()
        turtlebot.write_joint_state_to_sim(joint_pos, joint_vel)
        self._agent_heading = turtlebot.data.heading_w.reshape(-1).cpu().numpy()

        self.scene.reset()
        self.sim_context.step()

    def reset(self):
        """
        Resets the agent in either the top or bottom half of the map,
        and places the target in the opposite half.
        Coordinate system:
            x ∈ [-self.arena_size,self.arena_size]
            y ∈ [-self.arena_size,self.arena_size]   (origin at center)
        """
        # print("Resetting Env")

        # Randomly decide agent region (top or bottom)
        agent_on_top = np.random.rand() > 0.5

        # if agent_on_top:
        #     self._agent_position = self._sample_position(y_min=0, y_max=1, x_min=-1, x_max=1, margin=0)
        #     self._target_position = self._sample_position(y_min=-1, y_max=0, x_min=-1, x_max=1, margin=0)
        # else:
        #     self._agent_position = self._sample_position(y_min=-1, y_max=0, x_min=-1, x_max=1, margin=0)
        #     self._target_position = self._sample_position(y_min=0, y_max=1, x_min=-1, x_max=1, margin=0)
        # sample agent position until it is not colliding with any obstacle
        if agent_on_top:
            self._agent_position = self._sample_position(-0.2, self.arena_half, margin=0.2)
            self._target_position = self._sample_position(-self.arena_half, -self.arena_buffer_size)
        else:
            self._agent_position = self._sample_position(-self.arena_half, 0.2, margin=0.2)
            self._target_position = self._sample_position(self.arena_buffer_size, self.arena_half)

        self.reset_turtlebot()
        
        # print(f"Moved agent to {self._agent_position}")

        if self._config.num_obstacles:
            self.obstacles = self.build_obstacles_for_env(self._config.num_obstacles, self.obstacles_prim_path, self.arena_half, self.obstacle_size)
            self.move_obstacles(self.sim_dt, self.obstacles, self.arena_half, self._config.obstacle_size)

        target: RigidObject = self.scene["target"]
        target_pos = target.data.default_root_state.clone()
        target_pos[:, :2] = torch.tensor(self._target_position)
        target.write_root_pose_to_sim(target_pos[:, :7])

        # Reset timers
        self._current_time = 0.0
        self._lidar_readings = None
        self._lidar_measure_time = -1.0

        # Reset only environment physics (not entire sim)
        self.scene.reset()

        # Return initial observation
        return self.step(np.zeros(2), False)[0]

    def step(self, action: np.ndarray, render: bool):

        if self.sim_app.is_running():

            turtlebot: Articulation = self.scene["Turtlebot"]
            initial_position = turtlebot.data.root_pos_w.reshape(-1)[:2].cpu().numpy()

            if (int(100 * self._current_time) // 10) % (self._config.respawn_interval * 10) == 0:
                self._reset_target_location()

            scaled_action = action / 0.033
            turtlebot.set_joint_velocity_target(torch.tensor(scaled_action))
            self.scene.write_data_to_sim()

            #start_time = time.time()
            collision = False
            delta_time = 0
            for i in range(self.action_ratio):
                if i < self.action_ratio - 1:
                    self.sim_context.step(False)
                else:
                    self.sim_context.step(render)

                self._update_time()
                delta_time += self.sim_dt

                if self._config.num_obstacles:
                    self.move_obstacles(self.sim_dt, self.obstacles, self.arena_half, self._config.obstacle_size)

                if self._did_agent_collide():
                    collision = True
                    break

            self.scene.update(delta_time)
            # print(f"Sim took: {time.time() - start_time:.4f}")

            self._agent_position = turtlebot.data.root_pos_w.reshape(-1)[:2].cpu().numpy()
            self._agent_heading = turtlebot.data.heading_w.reshape(-1)[0].cpu().numpy()

            # print(f"Target Vector: {self._target_position}")
            # print(f"Distance: {np.linalg.norm(self._agent_position - self._target_position)}")

            # print(f"Agent starts at {initial_position}, moved to {self._agent_position}")

            reward = self._get_reward(initial_position, self._agent_position)
            # print(f"Reward: {reward}")

            lidar_readings: np.ndarray = self._get_lidar_readings()

            x_orientation = np.cos(self._agent_heading)
            y_orientation = np.sin(self._agent_heading)

            observation = {
                "agent_position": self._agent_position,
                "agent_orientation": np.array([x_orientation, y_orientation]),
                "agent_velocity": action,
                "target_position": self._get_noisy_target_position(),
                "lidar_readings": lidar_readings
            }

            done = collision or self._did_agent_collide() \
                or int(self._current_time // 1) > self._config.episode_length

            return observation, reward, done, {}
        
    def render_env(self):
        """Isaac Sim is already rendered"""
        pass

