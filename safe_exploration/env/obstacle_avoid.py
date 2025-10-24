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
        # Set the properties for spaces
        # cameron: the action space are velocity commands we send to the ball
        # It is one dimensional
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = Dict({
            'agent_position': Box(low=0, high=1, shape=(2,), dtype=np.float32),
            'target_position': Box(low=0, high=1, shape=(2,), dtype=np.float32),
            'lidar_readings': Box(low=0, high=1, shape=(8,), dtype=np.float32)
        })

        # Sets all the episode specific variables
        self._obstacles = np.array([
            [0.35, 0.35, 0.3, 0.3]
        ])
        dirs = np.array([
            [1, 0.01],
            [1, 1],
            [0.01, 1],
            [-1, 1],
            [-1, 0.01],
            [-1, -1],
            [0.01, -1],
            [1, -1],
        ])
        self._lidar_directions = dirs / np.expand_dims(np.linalg.norm(dirs, axis=1), axis=1)

        self._step_timestamp = time.time()
        self.reset()

        self.window = None
        self.clock = None
        self.window_size = 1024

    def point_in_boxes(self, point, box_set):
        return (point[0] >= box_set[:, 0]) & \
            (point[0] <= (box_set[:, 0] + box_set[:, 2])) & \
            (point[1] >= box_set[:, 1]) & \
            (point[1] <= (box_set[:, 1] + box_set[:, 3]))

    def raycast_horizontal_linesegments(self, ray_origin, ray_direction, line_set):
        # line_set is [y, x0, x1]
        assert ray_origin.shape == (2,)
        assert ray_direction.shape == (2,)
        assert line_set.shape[1] == 3

        y_diff = line_set[:, 0] - ray_origin[1]
        ray_x_at_y = ray_direction[0] / ray_direction[1] * y_diff + ray_origin[0]
        intersect_points = np.stack([ray_x_at_y, line_set[:, 0]], axis=1)
        
        difference = intersect_points - ray_origin
        distances = np.linalg.norm(difference, axis=1)
        in_ray_direction = np.all((ray_direction > 0) == (difference > 0), axis=1)
        
        intersects = ((ray_x_at_y >= line_set[:, 1]) & (ray_x_at_y <= line_set[:, 2])) | \
            ((ray_x_at_y >= line_set[:, 2]) & (ray_x_at_y <= line_set[:, 1]))
        valid = in_ray_direction & intersects
        
        distances[~valid] = 1.

        return np.min(distances)


    def raycast_vertical_linesegments(self, ray_origin, ray_direction, line_set):
        # line_set is [x, y0, y1]
        assert ray_origin.shape == (2,)
        assert ray_direction.shape == (2,)
        assert line_set.shape[1] == 3

        x_diff = line_set[:, 0] - ray_origin[0]
        ray_y_at_x = ray_direction[1] / ray_direction[0] * x_diff + ray_origin[1]
        intersect_points = np.stack([line_set[:, 0], ray_y_at_x], axis=1)
        
        difference = intersect_points - ray_origin
        distances = np.linalg.norm(difference, axis=1)
        in_ray_direction = np.all((ray_direction > 0) == (difference > 0), axis=1)
        
        intersects = ((ray_y_at_x >= line_set[:, 1]) & (ray_y_at_x <= line_set[:, 2])) | \
            ((ray_y_at_x >= line_set[:, 2]) & (ray_y_at_x <= line_set[:, 1]))
        valid = in_ray_direction & intersects
        
        distances[~valid] = 1.

        return np.min(distances)


    def raycast_boxes(self, ray_origin, ray_direction, box_set):
        horizontal_linesegments = np.concat(
            [
                np.stack([box_set[:, 1], box_set[:, 0], box_set[:, 0] + box_set[:, 2]], axis=1),
                np.stack([box_set[:, 1] + box_set[:, 3], box_set[:, 0], box_set[:, 0] + box_set[:, 2]], axis=1),
            ],
            axis=0
        )
        min_horizontal = self.raycast_horizontal_linesegments(ray_origin, ray_direction, horizontal_linesegments)

        vertical_linesegments = np.concat(
            [
                np.stack([box_set[:, 0], box_set[:, 1], box_set[:, 1] + box_set[:, 3]], axis=1),
                np.stack([box_set[:, 0] + box_set[:, 2], box_set[:, 1], box_set[:, 1] + box_set[:, 3]], axis=1),
            ],
            axis=0
        )
        min_vertical = self.raycast_vertical_linesegments(ray_origin, ray_direction, vertical_linesegments)

        return np.min([min_horizontal, min_vertical])
    
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
        agent_radians = np.random.random(1) * 2 * np.pi
        agent_distance = np.random.random(1) * 0.2 + 0.3
        self._agent_position = np.array([
            (agent_distance * np.cos(agent_radians))[0] + 0.5,
            (agent_distance * np.sin(agent_radians))[0] + 0.5
        ])

        obstacle_radians = agent_radians + np.pi + np.random.random(1) * 2 * np.pi * 0.10
        obstacle_distance = np.random.random(1) * 0.2 + 0.3
        self._target_position = np.array([
            (obstacle_distance * np.cos(obstacle_radians))[0] + 0.5,
            (obstacle_distance * np.sin(obstacle_radians))[0] + 0.5
        ])

        self._current_time = 0.

        self._lidar_readings = self._get_lidar_readings()

        return self.step(np.zeros(2))[0]
    
    def _reset_target_location(self):

        centered_agent_position = self._agent_position - 0.5
        agent_radians = np.arctan2(centered_agent_position[1], centered_agent_position[0])
        obstacle_radians = agent_radians + np.pi + ((np.random.random(1) - 0.5) * np.pi)
        obstacle_distance = np.random.random(1) * 0.2 + 0.3
        self._target_position = np.array([
            (obstacle_distance * np.cos(obstacle_radians))[0] + 0.5,
            (obstacle_distance * np.sin(obstacle_radians))[0] + 0.5
        ])
    
    def _get_reward(self, initial_position, final_position, action):
        if self._config.enable_reward_shaping and self._is_agent_outside_shaping_boundary():
            return -1
        else:
            initial_distance = LA.norm(initial_position - self._target_position)
            final_distance = LA.norm(final_position - self._target_position)
            distance_change = final_distance - initial_distance
            action_size = LA.norm(action)

            return -distance_change - 0.01 * action_size
    
    def _move_agent(self, velocity):
        # Assume that frequency of motor is 1 (one action per second)
        self._agent_position += self._config.frequency_ratio * velocity
    
    def _did_agent_collide(self):
        outside_boundary = np.any(self._agent_position < 0) or np.any(self._agent_position > 1)
        collide_with_obstacle = np.any(self.point_in_boxes(
            self._agent_position,
            self._obstacles
        ))

        return outside_boundary or collide_with_obstacle
    
    def _is_agent_outside_shaping_boundary(self):
        return np.any(self._agent_position < self._config.reward_shaping_slack) \
               or np.any(self._agent_position > 1 - self._config.reward_shaping_slack)

    def _update_time(self):
        # Assume that frequency of motor is 1 (one action per second)
        self._current_time += self._config.frequency_ratio
    
    def _get_noisy_target_position(self):
        return self._target_position + \
               np.random.normal(0, self._config.target_noise_std, 2)
    
    def get_num_constraints(self):
        # a max and a min in x and y
        return 2 * 2

    def get_constraint_values(self):
        # For any given n, there will be 2 * n constraints
        # a lower and upper bound for each dim
        # We define all the constraints such that C_i = 0
        # _agent_position > 0 + _agent_slack => -_agent_position + _agent_slack < 0
        min_constraints = self._config.agent_slack - self._agent_position
        # _agent_position < 1 - _agent_slack => _agent_position + agent_slack- 1 < 0
        max_constraint = self._agent_position + self._config.agent_slack - 1

        return np.concatenate([min_constraints, max_constraint])
    
    def _get_lidar_readings(self):
        number_of_lidars = self._lidar_directions.shape[0]
        agent_positions = np.tile(self._agent_position, reps=(number_of_lidars,1))
        return self.ray_aabb2d_distances(agent_positions, self._lidar_directions, self._obstacles)

        readings = np.ones(8)
        for lidar_idx in range(self._lidar_directions.shape[0]):
            lidar_direction = self._lidar_directions[lidar_idx]
            ray_reading = self.raycast_boxes(self._agent_position, lidar_direction, self._obstacles)
            readings[lidar_idx] = ray_reading

        return readings

    def step(self, action):
        # Check if the target needs to be relocated
        # Extract the first digit after decimal in current_time to add numerical stability
        if (int(100 * self._current_time) // 10) % (self._config.respawn_interval * 10) == 0:
            self._reset_target_location()

        # Increment time
        self._update_time()

        initial_position = self._agent_position.copy()
        # Calculate new position of the agent
        self._move_agent(action)

        # Find reward         
        reward = self._get_reward(initial_position, self._agent_position, action)

        self._lidar_readings = self._get_lidar_readings()

        # Prepare return payload
        observation = {
            "agent_position": self._agent_position,
            "target_postion": self._get_noisy_target_position(), # cameron: target position has noise,
            "lidar_readings": self._lidar_readings
        }

        done = self._did_agent_collide() \
               or int(self._current_time // 1) > self._config.episode_length

        return observation, reward, done, {}
    
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

        for obstacle_idx in range(self._obstacles.shape[0]):
            obstacle = self._obstacles[obstacle_idx]
            obstacle_screen_scale = obstacle * self.window_size

            pygame.draw.rect(
                canvas,
                (0, 255, 0),
                pygame.Rect(
                    (obstacle_screen_scale[0], obstacle_screen_scale[1]),
                    (obstacle_screen_scale[2], obstacle_screen_scale[3]),
                ),
            )

        for lidar_idx in range(self._lidar_directions.shape[0]):
            lidar_dir = self._lidar_directions[lidar_idx]
            lidar_dist = self._lidar_readings[lidar_idx]
            lidar_point = self._agent_position + lidar_dir * lidar_dist
            lidar_screen_point = lidar_point * self.window_size

            pygame.draw.line(
                canvas,
                (0, 0, 255),
                agent_screen_position,
                lidar_screen_point)

        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(10)
