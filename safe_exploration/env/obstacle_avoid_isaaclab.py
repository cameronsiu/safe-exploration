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
    from isaaclab.assets.articulation import Articulation
    from isaaclab.sensors import ContactSensor

from safe_exploration.core.config import Config

class ObstacleAvoidIsaacLab(gym.Env):

    def __init__(self, sim_app: SimulationApp, sim_context: SimulationContext, scene: InteractiveScene, pos: list):
        self._config = Config.get().env.obstacleavoidisaaclab
        self._action_scale = self._config.action_scale

        # NOTE: use IsaacSim to change the number of lidars and change the parameter in yaml
        self._num_lidar_buckets = self._config.num_lidar_buckets
        self._constraint_max_clip = self._config.constraint_max_clip

        # NOTE: turtlebot will apply velocity commands to wheel joints independently
        self.action_space = Box(low=-self._action_scale, high=self._action_scale, shape=(2,), dtype=np.float32)

        self.arena_half = self._config.arena_size // 2
        self.arena_buffer_size = self.arena_half*0.2

        self.observation_space = Dict({
            'agent_position': Box(low=-self.arena_half, high=self.arena_half, shape=(2,), dtype=np.float32),
            'agent_orientation': Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            #'agent_velocity': Box(low=-self._action_scale, high=self._action_scale, shape=(2,), dtype=np.float32),
            'target_position': Box(
                low=-(self.arena_half - self.arena_buffer_size),
                high=self.arena_half - self.arena_buffer_size,
                shape=(2,), 
                dtype=np.float32
            ),
            'lidar_readings': Box(low=0.2, high=100, shape=(self._num_lidar_buckets,), dtype=np.float32)
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
        obstacles_prim_path = f"/World/envs/env_0/Obstacles"
        if self._config.num_obstacles:
            self.obstacles = build_obstacles_for_env(self._config.num_obstacles, obstacles_prim_path, pos)
            self.move_obstacles = move_obstacles

        self.spawn_anywhere = False

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

    def _normalize_lidar_positions(self) -> np.ndarray:
        turtlebot: Articulation = self.scene["Turtlebot"]
        heading = turtlebot.data.heading_w.reshape(-1).cpu().numpy()[0]
        heading = heading % (2*np.pi)
        raw_lidar_readings: np.ndarray = self.lidar_interface.get_linear_depth_data(self.lidar_prim_path).reshape(-1)

        # amount of rotations per lidar ray
        angle_per_ray = 2 * np.pi / raw_lidar_readings.shape[0]
        lidar_shift = int(round(heading / angle_per_ray))
        raw_lidar_readings = np.roll(raw_lidar_readings, lidar_shift)
        return raw_lidar_readings

    def _get_lidar_readings(self) -> np.ndarray:
        if self._lidar_readings is not None and self._current_time == self._lidar_measure_time:
            return self._lidar_readings
        elif self._did_agent_collide():
            self._lidar_readings = np.zeros((self._num_lidar_buckets,))
            self._lidar_measure_time = self._current_time
            return self._lidar_readings
        else:
            raw_lidar_readings = self._normalize_lidar_positions()
            clipped_raw_lidar_readings = np.clip(raw_lidar_readings, 0.0, self._constraint_max_clip)
            bucket_size = clipped_raw_lidar_readings.shape[0] // self._num_lidar_buckets
            bucketed_lidar_readings = clipped_raw_lidar_readings.reshape((self._num_lidar_buckets, bucket_size))
            self._lidar_readings = np.min(bucketed_lidar_readings, axis=1)
            self._lidar_measure_time = self._current_time
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

    def get_num_constraints(self):
        return 1

    def get_constraint_values(self):
        clipped_readings = np.clip(self._get_lidar_readings(), 0, self._constraint_max_clip)
        return np.array([self._config.agent_slack - np.min(clipped_readings)])

    def reset(self):
        """
        Resets the agent in either the top or bottom half of the map,
        and places the target in the opposite half.
        Coordinate system:
            x ∈ [-self.arena_size,self.arena_size]
            y ∈ [-self.arena_size,self.arena_size]   (origin at center)
        """
        # print("Resetting Env")
        turtlebot: Articulation = self.scene["Turtlebot"]

        # Randomly decide agent region (top or bottom)
        agent_on_top = np.random.rand() > 0.5
        if agent_on_top:
            self._agent_position = self._sample_position(-0.2, self.arena_half, margin=0.2)
            self._target_position = self._sample_position(-self.arena_half, -self.arena_buffer_size)
        else:
            self._agent_position = self._sample_position(-self.arena_half, 0.2, margin=0.2)
            self._target_position = self._sample_position(self.arena_buffer_size, self.arena_half)

        if self.spawn_anywhere:
            self._agent_position = self._sample_position(x_min = -self.arena_half, x_max=self.arena_half, y_min = -self.arena_half, y_max=self.arena_half, margin=0.10)

        # print(self._target_position)
        heading = np.random.random() * 2 * np.pi
        q = torch.tensor([np.cos(heading / 2), 0, 0, np.sin(heading / 2)])

        # print(f"Moved agent to {self._agent_position}")

        # Write the agent pose into IsaacSim
        root_state = turtlebot.data.default_root_state.clone()
        root_state[:, :3] += self.scene.env_origins
        agent_position = torch.tensor(self._agent_position)
        root_state[:, 0:2] = agent_position
        root_state[:, 3:7] = q

        turtlebot.write_root_pose_to_sim(root_state[:, :7])
        turtlebot.write_root_velocity_to_sim(root_state[:, 7:])

        # Reset joints
        joint_pos = turtlebot.data.default_joint_pos.clone()
        joint_vel = turtlebot.data.default_joint_vel.clone()
        turtlebot.write_joint_state_to_sim(joint_pos, joint_vel)

        self._agent_heading = turtlebot.data.heading_w.reshape(-1).cpu().numpy()

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

        # HACK: Recursively call reset again so that agent doesn't spawn in obstacle
        self.sim_context.step()
        if self._did_agent_collide():
            return self.reset()

        # Return initial observation
        return self.step(np.zeros(2), False)[0]
    
    def normalize(self, vec):
        # Calculate the L2 norm
        l2_norm = np.linalg.norm(vec)

        # Normalize the vector
        unit_vec = vec / l2_norm

        return unit_vec
    
    def heading_to_wheel_vel(self, command_heading, current_heading_angle):
        max_wheel_velocity = 4.0
        command_magnitude = np.linalg.norm(command_heading)
        command_magnitude = np.clip(command_magnitude, 0., 1.)

        if command_magnitude < 0.01:
            return torch.tensor([[0.0, 0.0]])

        wheel_magnitude = max_wheel_velocity * command_magnitude
        command_direction = self.normalize(command_heading)

        command_angle = np.arctan2(command_direction[1], command_direction[0])
        angle_diff = command_angle - current_heading_angle

        rotation_threshold = 2.0 * np.pi / 16.0

        if np.abs(angle_diff) > rotation_threshold:
            # only rotate
            right_wheel_sign = np.sign(angle_diff)
            left_wheel_sign = -right_wheel_sign
            right_wheel_velocity = right_wheel_sign * wheel_magnitude
            left_wheel_velocity = left_wheel_sign * wheel_magnitude
            return torch.tensor([[left_wheel_velocity, right_wheel_velocity]])
        else:
            # rotate and drive
            rotation_fraction = angle_diff / rotation_threshold
            right_wheel_velocity = wheel_magnitude + rotation_fraction * 2.0
            left_wheel_velocity = wheel_magnitude - rotation_fraction * 2.0
            return torch.tensor([[left_wheel_velocity, right_wheel_velocity]])
        
    def normalize_angle(self, angle):
        single_rotation = angle % (2 * np.pi)
        
        if single_rotation > np.pi:
            normalized = single_rotation - 2 * np.pi
        elif single_rotation < -np.pi:
            normalized = single_rotation - -2 * np.pi
        else:
            normalized = single_rotation

        return normalized
        
    def waypoint_to_wheel_vec(self, goal_pos, current_pos, current_heading_angle):
        diff = goal_pos - current_pos

        #print(f"goal_position: {goal_pos}")
        #print(f"current_pos: {current_pos}")

        diff_magnitude = np.linalg.norm(diff)

        if diff_magnitude < 0.001:
            return torch.tensor([[0.0, 0.0]])

        diff_norm = self.normalize(diff)
        command_angle = np.arctan2(diff_norm[1], diff_norm[0])
        forward_angle_diff = self.normalize_angle(command_angle - current_heading_angle)
        reverse_angle_diff = self.normalize_angle(command_angle - (current_heading_angle + np.pi))

        if np.abs(forward_angle_diff) > np.abs(reverse_angle_diff):
            angle_diff = reverse_angle_diff
            velocity_mult = -1
        else:
            angle_diff = forward_angle_diff
            velocity_mult = 1

        #print(f"command_angle: {command_angle}")
        #print(f"angle_diff: {angle_diff}")

        rotation_threshold = 2.0 * np.pi / 16.0

        wheel_speed = 6.0
        if np.abs(angle_diff) > rotation_threshold:
            # only rotate
            right_wheel_sign = np.sign(angle_diff)
            left_wheel_sign = -right_wheel_sign
            right_wheel_velocity = right_wheel_sign * wheel_speed
            left_wheel_velocity = left_wheel_sign * wheel_speed
            return velocity_mult * torch.tensor([[left_wheel_velocity, right_wheel_velocity]])
        else:
            # rotate and drive
            rotation_fraction = angle_diff / rotation_threshold
            rotation_adjustment_wheel_velocity = 4.0
            if rotation_fraction > 0:
                right_wheel_velocity = wheel_speed
                left_wheel_velocity = wheel_speed - rotation_fraction * rotation_adjustment_wheel_velocity
            else:
                right_wheel_velocity = wheel_speed + rotation_fraction * rotation_adjustment_wheel_velocity
                left_wheel_velocity = wheel_speed

            return velocity_mult * torch.tensor([[left_wheel_velocity, right_wheel_velocity]])


    def step(self, action: np.ndarray, render: bool):

        if self.sim_app.is_running():

            turtlebot: Articulation = self.scene["Turtlebot"]
            initial_position = turtlebot.data.root_pos_w.reshape(-1)[:2].cpu().numpy()

            goal_position = action + self._agent_position

            #print(f"action: {action}")

            done = False
            reached_goal = False
            action_steps = 0
            while not done:
                action_steps += 1

                wheel_velocity = self.waypoint_to_wheel_vec(goal_position, self._agent_position, self._agent_heading)
                turtlebot.set_joint_velocity_target(wheel_velocity)
                self.scene.write_data_to_sim()

                if reached_goal:
                    self.sim_context.step(render)
                else:
                    self.sim_context.step(render)

                self._update_time()

                if self._config.num_obstacles:
                    self.move_obstacles(self.sim_dt, self.obstacles, self.arena_half, self._config.obstacle_size)

                self.scene.update(self.sim_dt)
                self._agent_position = turtlebot.data.root_pos_w.reshape(-1)[:2].cpu().numpy()
                self._agent_heading = turtlebot.data.heading_w.reshape(-1)[0].cpu().numpy()

                done = self._did_agent_collide() \
                    or int(self._current_time // 1) > self._config.episode_length
                
                if (int(100 * self._current_time) // 10) % (self._config.respawn_interval * 10) == 0:
                    self._reset_target_location()

                if reached_goal:
                    break

                if np.linalg.norm(goal_position - self._agent_position) < 0.005:
                    reached_goal = True

            #print(action_steps)
            #print(np.linalg.norm(self._agent_position - initial_position))

            reward = self._get_reward(initial_position, self._agent_position)

            lidar_readings: np.ndarray = self._get_lidar_readings()

            x_orientation = np.cos(self._agent_heading)
            y_orientation = np.sin(self._agent_heading)

            observation = {
                "agent_position": self._agent_position,
                "agent_orientation": np.array([x_orientation, y_orientation]),
                #"agent_velocity": action / 0.033,
                "target_position": self._get_noisy_target_position(),
                "lidar_readings": lidar_readings
            }


            return observation, reward, done, { "wheel_velocity": wheel_velocity }
        
    def render_env(self):
        """Isaac Sim is already rendered"""
        pass

