import gym
from gym.spaces import Box, Dict
import numpy as np
from numpy import linalg as LA
import pygame
import time

from safe_exploration.core.config import Config


class ObstacleAvoid(gym.Env):
    def __init__(self):
        self._config = Config.get().env.obstacleavoid
        self._action_scale = self._config.action_scale

        self._num_lidar_buckets = 12
        num_lidars = self._num_lidar_buckets * 4
        self._lidar_directions = self._make_lidar_directions(num_lidars)

        self.action_space = Box(low=-self._action_scale, high=self._action_scale, shape=(2,), dtype=np.float32)
        self.observation_space = Dict({
            'agent_position': Box(low=0, high=1, shape=(2,), dtype=np.float32),
            'target_position': Box(low=0, high=1, shape=(2,), dtype=np.float32),
            #'nearest_lidar_distance': Box(low=0, high=1, shape=(1,), dtype=np.float32),
            #'nearest_lidar_direction': Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            #'lidar_readings_x': Box(low=0, high=1, shape=(num_lidars,), dtype=np.float32),
            #'lidar_readings_y': Box(low=0, high=1, shape=(num_lidars,), dtype=np.float32),
            'lidar_readings': Box(low=0, high=1, shape=(self._num_lidar_buckets,), dtype=np.float32)
        })

        # Sets all the episode specific variables
        # X, Y, W, H
        self._fixed_obstacles = np.array([
            # Walls around border
            [-1, 0, 1.02, 1],
            [0.98, 0, 1, 1],
            [-1, -1, 3, 1.02],
            [-1, 0.98, 3, 1],
        ])

        self._changing_obstacles = np.array([])
        self._moving_obstacles = np.array([])
        self._moving_obstacle_directions = []

        self._step_timestamp = time.time()
        self.reset()

        self.window = None
        self.clock = None
        self.window_size = 1024

    def _get_obstacles(self):
        return np.concat([self._fixed_obstacles, self._changing_obstacles, self._moving_obstacles], axis=0)

    def _make_lidar_directions(self, number_of_rays):
        lidar_directions = np.zeros((number_of_rays, 2))
        spacing = 2 * np.pi / number_of_rays
        for i in range(number_of_rays):
            # add 0.001 so there aren't flat/vertical lines
            angle = i * spacing + 0.001
            x = np.cos(angle)
            y = np.sin(angle)
            lidar_directions[i] = np.array([x, y])

        return lidar_directions


    def point_in_boxes(self, point, box_set):
        return (point[0] >= box_set[:, 0]) & \
            (point[0] <= (box_set[:, 0] + box_set[:, 2])) & \
            (point[1] >= box_set[:, 1]) & \
            (point[1] <= (box_set[:, 1] + box_set[:, 3]))
    
    
    def ray_aabb2d_distances(
        self,
        origins: np.ndarray,      # (N, 2)
        directions: np.ndarray,   # (N, 2) -- assumed normalized
        boxes: np.ndarray,        # (M, 4) -> [x, y, width, height]
        eps: float = 1e-12,
    ) -> np.ndarray:
        """
        Compute distances (t in [0, 1]) to the first AABB hit for each ray.
        Returns 1.0 if the ray does not hit any box within [0,1].

        Parameters
        ----------
        origins : (N, 2)
            Ray origins.
        directions : (N, 2)
            Normalized ray directions.
        boxes : (M, 4)
            [x, y, width, height] for each AABB (bottom-left corner).
        eps : float
            Tolerance for zero direction components.

        Returns
        -------
        distances : (N,)
            Distance to the first hit (or 1.0 if none).
        """
        # Extract mins and maxs
        Bmin = boxes[:, :2]                    # (M,2)
        Bmax = boxes[:, :2] + boxes[:, 2:]     # (M,2)

        O = origins[:, None, :]     # (N,1,2)
        D = directions[:, None, :]  # (N,1,2)
        Bmin = Bmin[None, :, :]     # (1,M,2)
        Bmax = Bmax[None, :, :]     # (1,M,2)

        # Safe reciprocal of direction
        dir_zero = np.abs(D) <= eps
        invD = np.empty_like(D)
        invD[:] = 0.0
        invD[~dir_zero] = 1.0 / D[~dir_zero]

        # Slab intersections
        t0 = (Bmin - O) * invD
        t1 = (Bmax - O) * invD
        tmin_axis = np.minimum(t0, t1)
        tmax_axis = np.maximum(t0, t1)

        # Handle parallel axes
        Ocmp = O.repeat(Bmin.shape[1], axis=1)
        inside_axis = (Ocmp >= Bmin) & (Ocmp <= Bmax)
        tmin_axis = np.where(dir_zero &  inside_axis, -np.inf, tmin_axis)
        tmax_axis = np.where(dir_zero &  inside_axis, +np.inf, tmax_axis)
        tmin_axis = np.where(dir_zero & ~inside_axis, +np.inf, tmin_axis)
        tmax_axis = np.where(dir_zero & ~inside_axis, -np.inf, tmax_axis)

        # Aggregate over x/y
        t_enter = np.max(tmin_axis, axis=2)  # (N,M)
        t_exit  = np.min(tmax_axis, axis=2)  # (N,M)

        # Valid unit-length hits
        hits = (t_exit >= t_enter) & (t_enter >= 0.0) & (t_enter <= 1.0)

        # Nearest hit per ray
        nearest_t = np.where(hits, t_enter, np.inf)
        best_t = np.min(nearest_t, axis=1)
        distances = np.where(np.isfinite(best_t), best_t, 1.0)

        return distances
        
        
    def reset(self):
        
        self._reset_obstacle_positions()
        self._reset_moving_obstacle_positions()
        self._reset_agent_position()
        while self._did_agent_collide():
            self._reset_agent_position()

        self._reset_target_location()

        self._current_time = 0.

        self._lidar_measure_time = -1.0
        self._lidar_readings = None

        return self.step(np.zeros(2), False)[0]
    
    def _reset_obstacle_positions(self):
        num_obstacles = 20
        scale = np.tile([0.8, 0.8], (num_obstacles, 1))
        origin = np.tile([0.1, 0.1], (num_obstacles, 1))
        positions = np.random.random((num_obstacles, 2)) * scale + origin
        sizes = np.tile([0.05, 0.05], (num_obstacles, 1))
        self._changing_obstacles = np.concat([positions, sizes], axis=1)

    def _reset_moving_obstacle_positions(self):
        num_obstacles = 5
        scale = np.tile([0.8, 0.8], (num_obstacles, 1))
        origin = np.tile([0.1, 0.1], (num_obstacles, 1))
        positions = np.random.random((num_obstacles, 2)) * scale + origin
        sizes = np.tile([0.05, 0.05], (num_obstacles, 1))
        self._moving_obstacles = np.concat([positions, sizes], axis=1)
        self._moving_obstacle_directions = [1] * num_obstacles

    def _reset_agent_position(self):
        agent_position = np.random.random(2)
        agent_on_top = np.random.random(1)[0] > 0.5

        if agent_on_top:
            self._agent_position = np.array([0.1, 0.9]) + agent_position * np.array([0.8, -0.15])
        else:
            self._agent_position = np.array([0.1, 0.1]) + agent_position * np.array([0.8, 0.15])
    
    def _reset_target_location(self):
        agent_on_top = self._agent_position[1] > 0.5
        target_position = np.random.random(2)

        if agent_on_top:
            self._target_position = np.array([0.1, 0.1]) + target_position * np.array([0.8, 0.15])
        else:
            self._target_position = np.array([0.1, 0.8]) + target_position * np.array([0.8, -0.15])
    
    def _get_reward(self, initial_position, final_position, action, lidar_readings, collided):
        if self._config.enable_reward_shaping and self._is_agent_outside_shaping_boundary():
            return -1, -1
        else:
            # Square each distance so being closer is more valuable
            sq_initial_distance = (1 - LA.norm(initial_position - self._target_position)) ** 2
            sq_final_distance = (1 - LA.norm(final_position - self._target_position)) ** 2
            distance_change = sq_final_distance - sq_initial_distance

            lidar_signal = np.min(lidar_readings) * 0.05

            if collided:
                collision_signal = -1
            else:
                collision_signal = 0

            #reward = distance_change + lidar_signal + collision_signal
            reward = distance_change

            return reward, distance_change
    
    def _move_agent(self, velocity):
        # Assume that frequency of motor is 1 (one action per second)
        self._agent_position = self._agent_position + self._config.frequency_ratio * velocity
    
    def _did_agent_collide(self):
        collide_with_obstacle = np.any(self.point_in_boxes(
            self._agent_position,
            self._get_obstacles()
        ))

        return collide_with_obstacle
    
    def _is_agent_outside_shaping_boundary(self):
        return np.any(self._get_lidar_readings() < self._config.reward_shaping_slack)

    def _update_time(self):
        # Assume that frequency of motor is 1 (one action per second)
        self._current_time += self._config.frequency_ratio
    
    def _get_noisy_target_position(self):
        return self._target_position + \
               np.random.normal(0, self._config.target_noise_std, 2)
    
    def get_num_constraints(self):
        # return self._lidar_directions.shape[0]
        return 1

    def get_constraint_values(self):
        # return self._config.agent_slack - self._get_lidar_readings()
        clipped_readings = np.clip(self._get_lidar_readings(), 0, 0.2)
        return np.array([self._config.agent_slack - np.min(clipped_readings)])
    
    def _get_lidar_readings(self):
        if self._lidar_readings is not None and self._current_time == self._lidar_measure_time:
            return self._lidar_readings
        elif self._did_agent_collide():
            self._lidar_readings = np.zeros((self._num_lidar_buckets))
            self._lidar_measure_time = self._current_time
            return self._lidar_readings
        else:
            number_of_lidars = self._lidar_directions.shape[0]
            agent_positions = np.tile(self._agent_position, reps=(number_of_lidars,1))
            lidar_distances = self.ray_aabb2d_distances(agent_positions, self._lidar_directions, self._get_obstacles())
            distances_per_bucket = int(lidar_distances.shape[0] / self._num_lidar_buckets)
            self._lidar_readings = np.array([np.min(lidar_distances[i:i+distances_per_bucket]) for i in range(0, lidar_distances.shape[0], distances_per_bucket)])
            self._lidar_measure_time = self._current_time
            return self._lidar_readings
        
    def _move_obstacles(self):
        for obstacle_idx in range(self._moving_obstacles.shape[0]):
            obstacle_y = self._moving_obstacles[obstacle_idx, 1]
            obstacle_direction = self._moving_obstacle_directions[obstacle_idx]

            y_diff = self._config.frequency_ratio * obstacle_direction * 0.1
            new_y = obstacle_y + y_diff
            self._moving_obstacles[obstacle_idx, 1] = new_y

            if (new_y > 0.85 and obstacle_direction > 0) or (new_y < 0.15 and obstacle_direction < 0):
                self._moving_obstacle_directions[obstacle_idx] = -1 * obstacle_direction

    def step(self, action, render):
        # Check if the target needs to be relocated
        # Extract the first digit after decimal in current_time to add numerical stability
        if (int(100 * self._current_time) // 10) % (self._config.respawn_interval * 10) == 0:
            self._reset_target_location()

        # Increment time
        self._update_time()

        initial_position = self._agent_position.copy()
        # Calculate new position of the agent
        self._move_agent(action)
        self._move_obstacles()

        lidar_readings = self._get_lidar_readings()
        # nearest_lidar = np.argmin(lidar_readings)
        # Prepare return payload
        observation = {
            "agent_position": self._agent_position,
            "target_position": self._get_noisy_target_position(),
            #"nearest_lidar_distance": lidar_readings[nearest_lidar:nearest_lidar+1],
            #"nearest_lidar_direction": self._lidar_directions[nearest_lidar],
            #"lidar_readings_x": scaled_lidar_directions[:, 0],
            #"lidar_readings_y": scaled_lidar_directions[:, 1],
            "lidar_readings": lidar_readings
        }

        collided = self._did_agent_collide()
        done = collided \
               or int(self._current_time // 1) > self._config.episode_length

        # Find reward
        reward, old_reward = self._get_reward(initial_position, self._agent_position, action, lidar_readings, collided)

        return observation, reward, done, { "old_reward": old_reward }
    
    def render_env(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = 50

        # First we draw the target
        target_screen_position = self.window_size * self._target_position
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            [int(target_screen_position[0]), int(target_screen_position[1])],
            10,
        )

        # Now we draw the agent
        agent_screen_position = self._agent_position * self.window_size
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            [int(agent_screen_position[0]), int(agent_screen_position[1])],
            10,
        )

        obstacles = self._get_obstacles()
        for obstacle_idx in range(obstacles.shape[0]):
            obstacle = obstacles[obstacle_idx]
            obstacle_screen_scale = obstacle * self.window_size

            pygame.draw.rect(
                canvas,
                (0, 255, 0),
                pygame.Rect(
                    (obstacle_screen_scale[0], obstacle_screen_scale[1]),
                    (obstacle_screen_scale[2], obstacle_screen_scale[3]),
                ),
            )

        lidar_readings = self._get_lidar_readings()
        num_lidars = lidar_readings.shape[0]
        lidar_directions = self._make_lidar_directions(num_lidars)
        for lidar_idx in range(num_lidars):
            lidar_dir = lidar_directions[lidar_idx]
            lidar_dist = lidar_readings[lidar_idx]
            lidar_point = self._agent_position + lidar_dir * lidar_dist
            lidar_screen_point = lidar_point * self.window_size

            pygame.draw.line(
                canvas,
                (0, 0, 255, 100),
                agent_screen_position,
                lidar_screen_point)

        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(10)
