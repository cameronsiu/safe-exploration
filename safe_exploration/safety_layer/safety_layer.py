from datetime import datetime
from functional import seq
from typing import List
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from safe_exploration.core.config import Config
from safe_exploration.core.replay_buffer import ReplayBuffer
from safe_exploration.core.tensorboard import TensorBoard
from safe_exploration.safety_layer.constraint_model import ConstraintModel
from safe_exploration.utils.list import for_each

class SafetyLayer:
    def __init__(self, env, constraint_model_files:List[str], render: bool=False):    
        self._env = env
        self._config = Config.get().safety_layer.trainer

        self._num_constraints = env.get_num_constraints()
        if self._num_constraints != len(constraint_model_files):
            constraint_model_files = [None]*self._num_constraints

        self._features = ["lidar_readings"]

        self._initialize_constraint_models(constraint_model_files)

        self._replay_buffer = ReplayBuffer(self._config.replay_buffer_size)

        # Tensorboard writer
        self._writer = TensorBoard.get_writer()
        self._train_global_step = 0
        self._eval_global_step = 0

        if self._config.use_gpu:
            self._cuda()

        self._render = render

        self.collisions = 0

        if self._config.save_data:
            self.save_data = {
                "action": [],
                "observation": [],
                "c": [],
                "c_next": [],
                "agent_position": [],
                "collided": []
            }

    def _as_tensor(self, ndarray, requires_grad=False):
        tensor = torch.Tensor(ndarray)
        tensor.requires_grad = requires_grad
        return tensor

    def _cuda(self):
        for_each(lambda x: x.cuda(), self._models)

    def _eval_mode(self):
        for_each(lambda x: x.eval(), self._models)

    def _train_mode(self):
        for_each(lambda x: x.train(), self._models)

    def _flatten_dict(self, inp):
        if type(inp) == dict:
            inp = np.concatenate(list([inp[key] for key in sorted(inp.keys())]))
        return inp

    def _initialize_constraint_models(self, constraint_model_files: List[str]):
        feature_dict = {
            feature: self._env.observation_space.spaces[feature] for feature in self._features
        }
        observation_dim = (seq(feature_dict.values())
                            .map(lambda x: x.shape[0])
                            .sum())
        #observation_dim = 1
        self._models = [ConstraintModel(observation_dim,
                                        self._env.action_space.shape[0], model_file) \
                        for model_file in constraint_model_files]

        self._optimizers = [Adam(x.parameters(), lr=self._config.lr) for x in self._models]

    """
    Experimental:
    def _sample_steps(self, num_steps):
    episode_length = 0
    observation = self._env.reset()
    self.collisions = 0

    # --- NEW: Define a behavior mode ---
    behavior_step_remaining = 0
    behavior_type = None
    left_vel = right_vel = 0.0

    def sample_new_behavior():
        '''Return new behavior_type and number of steps it will last.'''
        behavior = np.random.choice([
            "forward",
            "backward",
            "turn_forward",
            "turn_backward",
            "spin",
            "wiggle",
            "pause"
        ], p=[0.25, 0.1, 0.25, 0.1, 0.1, 0.15, 0.05])  # tune probabilities

        duration = np.random.randint(20, 80)  # how long this behavior lasts
        return behavior, duration

    # sample first behavior
    behavior_type, behavior_step_remaining = sample_new_behavior()

    for step in range(num_steps):

        # ---------- SWITCH BEHAVIOR WHEN DONE ----------
        if behavior_step_remaining <= 0:
            behavior_type, behavior_step_remaining = sample_new_behavior()

        behavior_step_remaining -= 1

        # ---------- BEHAVIOR GENERATION ----------
        if behavior_type == "forward":
            base = np.random.uniform(0.05, 0.22)
            turn = np.random.uniform(-0.05, 0.05)
            left_vel  = base - turn
            right_vel = base + turn

        elif behavior_type == "backward":
            base = np.random.uniform(-0.22, -0.05)
            turn = np.random.uniform(-0.05, 0.05)
            left_vel  = base - turn
            right_vel = base + turn

        elif behavior_type == "turn_forward":
            base = np.random.uniform(0.05, 0.22)
            turn = np.random.uniform(0.05, 0.18) * np.random.choice([-1, 1])
            left_vel  = np.clip(base - turn, -0.22, 0.22)
            right_vel = np.clip(base + turn, -0.22, 0.22)

        elif behavior_type == "turn_backward":
            base = np.random.uniform(-0.22, -0.05)
            turn = np.random.uniform(0.05, 0.18) * np.random.choice([-1, 1])
            left_vel  = np.clip(base - turn, -0.22, 0.22)
            right_vel = np.clip(base + turn, -0.22, 0.22)

        elif behavior_type == "spin":
            spin_dir = np.random.choice([-1, 1])
            left_vel  = -spin_dir * np.random.uniform(0.12, 0.22)
            right_vel =  spin_dir * np.random.uniform(0.12, 0.22)

        elif behavior_type == "wiggle":
            base = np.random.uniform(0.05, 0.15)
            turn = np.sin(step * 0.3) * np.random.uniform(0.05, 0.15)
            left_vel  = np.clip(base - turn, -0.22, 0.22)
            right_vel = np.clip(base + turn, -0.22, 0.22)

        elif behavior_type == "pause":
            left_vel = right_vel = 0.0

        action = np.array([left_vel, right_vel])

        # ------------------------------------------------

        c = self._env.get_constraint_values()
        observation_next, _, done, _ = self._env.step(action, self._render)
        c_next = self._env.get_constraint_values()

        if self._render:
            self._env.render_env()

        self._replay_buffer.add({
            "action": action,
            "observation": self._flatten_dict({
                feature: observation[feature] for feature in self._features
            }),
            "c": c,
            "c_next": c_next
        })

        if self._config.save_data:
            self.save_data["action"].append(action)
            self.save_data["observation"].append(self._flatten_dict({
                feature: observation[feature] for feature in self._features
            }))
            self.save_data["c"].append(c)
            self.save_data["c_next"].append(c_next)
            self.save_data["agent_position"].append(observation["agent_position"])

        observation = observation_next
        episode_length += 1

        if self._env._did_agent_collide():
            self.collisions += 1

        if done or (episode_length == self._config.max_episode_length):
            observation = self._env.reset()
            episode_length = 0

            # sample new behavior next episode
            behavior_type, behavior_step_remaining = sample_new_behavior()
    """

    def _sample_steps(self, num_steps):
        self._env.spawn_anywhere = True
        episode_length = 0
        observation = self._env.reset()
        self.collisions = 0
        self.constraint_violations = 0

        for step in range(num_steps):
            # Forward speed bias
            # bias_speed = (np.random.random() * 0.2) + 0.8
            # Random turning bias
            # turn_bias = (np.random.random() * 0.12) - 0.06

            # base = 0.22 * bias_speed
            # left_vel  = np.clip(base - turn_bias, -0.22, 0.22)
            # right_vel = np.clip(base + turn_bias, -0.22, 0.22)
            # action = np.array([left_vel, right_vel])
            if step % 100 == 0:
                action = self._env.action_space.sample()
            c = self._env.get_constraint_values()
            observation_next, _, done, _ = self._env.step(action, self._render)
            c_next = self._env.get_constraint_values()

            if self._render:
                self._env.render_env()

            self._replay_buffer.add({
                "action": action,
                "observation": self._flatten_dict({
                    feature: observation[feature] for feature in self._features
                }),
                "c": c,
                "c_next": c_next
            })

            collided = self._env._did_agent_collide()

            if self._config.save_data:
                self.save_data["action"].append(action)
                self.save_data["observation"].append(self._flatten_dict({
                    feature: observation[feature] for feature in self._features
                }))
                self.save_data["c"].append(c)
                self.save_data["c_next"].append(c_next)
                self.save_data["agent_position"].append(observation["agent_position"])
                self.save_data["collided"].append(collided)
            
            observation = observation_next
            episode_length += 1

            if collided:
                self.collisions += 1

            if np.any(c > 0):
                self.constraint_violations += 1
            
            if done or (episode_length == self._config.max_episode_length):
                observation = self._env.reset()
                episode_length = 0
        
        self._env.spawn_anywhere = False

    def _evaluate_batch(self, batch):
        observation = self._as_tensor(batch["observation"])
        action = self._as_tensor(batch["action"])
        c = self._as_tensor(batch["c"])
        c_next = self._as_tensor(batch["c_next"])
        
        gs = [x(observation) for i, x in enumerate(self._models)]

        c_next_predicted = [c[:, i] + \
                            torch.bmm(x.view(x.shape[0], 1, -1), action.view(action.shape[0], -1, 1)).view(-1) \
                            for i, x in enumerate(gs)]
        losses = [torch.mean((c_next[:, i] - c_next_predicted[i]) ** 2) for i in range(self._num_constraints)]

        return losses

    def _update_batch(self, batch):
        batch = self._replay_buffer.sample(self._config.batch_size)

        # Update critic
        for_each(lambda x: x.zero_grad(), self._optimizers)
        losses = self._evaluate_batch(batch)
        for_each(lambda x: x.backward(), losses)
        for_each(lambda x: x.step(), self._optimizers)

        return np.asarray([x.item() for x in losses])

    def evaluate(self):
        # Sample steps
        self._sample_steps(self._config.evaluation_steps)

        self._eval_mode()
        # compute losses
        losses = [list(map(lambda x: x.item(), self._evaluate_batch(batch))) for batch in \
                self._replay_buffer.get_sequential(self._config.batch_size)]

        losses = np.mean(np.concatenate(losses).reshape(-1, self._num_constraints), axis=0)

        self._replay_buffer.clear()
        # Log to tensorboard
        for_each(lambda x: self._writer.add_scalar(f"constraint {x[0]} eval loss", x[1], self._eval_global_step),
                 enumerate(losses))
        self._eval_global_step += 1

        self._train_mode()

        print(f"Validation completed, average loss {losses}\n"
              f"Number of collisions: {self.collisions}")

    def get_safe_action_with_details(self, observation, action, c):    
        # Find the values of G
        self._eval_mode()
        o = self._as_tensor(self._flatten_dict({
            feature: observation[feature] for feature in self._features
        })).view(1, -1)
        #g = [x(o[:, i:i+1]) for i, x in enumerate(self._models)]
        g = [x(o) for i, x in enumerate(self._models)]
        self._train_mode()

        # Find the lagrange multipliers
        g = [x.data.numpy().reshape(-1) for x in g]

        unclipped_multipliers = [(np.dot(g_i, action) + c_i) / np.dot(g_i, g_i) for g_i, c_i in zip(g, c)]
        multipliers = [np.clip(x, 0, np.inf) for x in unclipped_multipliers]

        # Calculate correction
        correction = np.max(multipliers) * g[np.argmax(multipliers)]

        action_new = action - correction

        action_clipped = np.clip(action_new, -self._env._action_scale, self._env._action_scale)

        details = {
            "g": g,
            "unclipped_multipliers": unclipped_multipliers,
            "multipliers": multipliers,
            "correction": correction,
            "action_new": action_new
        }

        return action_clipped, details

    def get_safe_action(self, observation, action, c):    
        action_clipped, details = self.get_safe_action_with_details(observation, action, c)
        return action_clipped

    def train(self, output_folder:str):

        start_time = time.time()

        print("==========================================================")
        print("Initializing constraint model training...")
        print("----------------------------------------------------------")        
        print(f"Start time: {datetime.fromtimestamp(start_time)}")
        print("==========================================================")

        print(f"Safety Layer Tensorboard folder: {self._writer.logdir}")

        number_of_steps = self._config.steps_per_epoch * self._config.epochs

        for epoch in range(self._config.epochs):
            # Just sample episodes for the whole epoch
            self._sample_steps(self._config.steps_per_epoch)

            # Do the update from memory
            losses = np.mean(np.concatenate([self._update_batch(batch) for batch in \
                    self._replay_buffer.get_sequential(self._config.batch_size)]).reshape(-1, self._num_constraints), axis=0)

            self._replay_buffer.clear()

            # Write losses and histograms to tensorboard
            for_each(lambda x: self._writer.add_scalar(f"constraint {x[0]} training loss", x[1], self._train_global_step),
                     enumerate(losses))

            (seq(self._models)
                    .zip_with_index() # (model, index) 
                    .map(lambda x: (f"constraint_model_{x[1]}", x[0])) # (model_name, model)
                    .flat_map(lambda x: [(x[0], y) for y in x[1].named_parameters()]) # (model_name, (param_name, param_data))
                    .map(lambda x: (f"{x[0]}_{x[1][0]}", x[1][1])) # (modified_param_name, param_data)
                    .for_each(lambda x: self._writer.add_histogram(x[0], x[1].data.numpy(), self._train_global_step)))

            self._train_global_step += 1

            print(f"Finished epoch {epoch} with losses: {losses}.\n"
                  f"Number of collisions: {self.collisions}\n"
                  f"Number of constraint violations: {self.constraint_violations}\n"
                  f"Running validation ...")
            self.evaluate()
            print("----------------------------------------------------------")

            for i, model in enumerate(self._models):
                model.save(output_folder, i)

            if self._config.save_data:
                self.save_replay_buffer()

        self._writer.close()
        print("==========================================================")
        print(f"Finished training constraint model. Time spent: {(time.time() - start_time) // 1} secs")
        print("==========================================================")

        for i, model in enumerate(self._models):
            model.save(output_folder, i)

        if self._config.save_data:
            self.save_replay_buffer()

    def save_replay_buffer(self, filename="data/safety_layer/replay_buffer.npz"):
        actions = np.array(self.save_data["action"])
        observations = np.array(self.save_data["observation"])
        c = np.array(self.save_data["c"])
        c_next = np.array(self.save_data["c_next"])
        agent_position = np.array(self.save_data["agent_position"])
        collided = np.array(self.save_data["collided"])

        print(actions.shape)
        print(observations.shape)
        print(c.shape)
        print(c_next.shape)
        print(agent_position.shape)

        np.savez_compressed(filename,
                            actions=actions,
                            observations=observations,
                            c=c,
                            c_next=c_next,
                            agent_position=agent_position,
                            collided=collided
        )
        print(f"Data saved to {filename}")

