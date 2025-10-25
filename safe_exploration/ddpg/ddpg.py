import copy
from datetime import datetime
from functional import seq
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam

from safe_exploration.core.config import Config
from safe_exploration.core.replay_buffer import ReplayBuffer
from safe_exploration.core.tensorboard import TensorBoard
from safe_exploration.utils.list import for_each, select_with_predicate

class DDPG:
    def __init__(self,
                 env,
                 actor,
                 critic,
                 action_modifier=None,
                 render_training=False,
                 render_evaluation=False):
        self._env = env
        self._actor = actor
        self._critic = critic
        self._action_modifier = action_modifier

        self._config = Config.get().ddpg.trainer

        self._initialize_target_networks()
        self._initialize_optimizers()

        self._models = {
            'actor': self._actor,
            'critic': self._critic,
            'target_actor': self._target_actor,
            'target_critic': self._target_critic
        }

        self._replay_buffer = ReplayBuffer(self._config.replay_buffer_size)

        # Tensorboard writer
        self._writer = TensorBoard.get_writer()
        self._train_global_step = 0
        self._eval_global_step = 0

        if self._config.use_gpu:
            self._cuda()

        self._render_training = render_training
        self._render_evaluation = render_evaluation

        self._batch_sample_time = 0
        self._tensor_convert_time = 0
        self._actor_update_time = 0
        self._critic_update_time = 0
        self._logging_time = 0
        self._target_compute_time = 0
        self._target_copy_time = 0

    def _as_tensor(self, ndarray, requires_grad=False):
        tensor = torch.Tensor(ndarray)
        tensor.requires_grad = requires_grad
        return tensor

    def _initialize_target_networks(self):
        self._target_actor = copy.deepcopy(self._actor)
        self._target_critic = copy.deepcopy(self._critic)
    
    def _initialize_optimizers(self):
        self._actor_optimizer = Adam(self._actor.parameters(), lr=self._config.actor_lr)
        self._critic_optimizer = Adam(self._critic.parameters(), lr=self._config.critic_lr)
    
    def _eval_mode(self):
        for_each(lambda x: x.eval(), self._models.values())

    def _train_mode(self):
        for_each(lambda x: x.train(), self._models.values())

    def _cuda(self):
        for_each(lambda x: x.cuda(), self._models.values())

    def _get_action(self, observation, c, is_training=True):
        # Action + random gaussian noise (as recommended in spining up)
        action = self._actor(self._as_tensor(self._flatten_dict(observation)))
        if is_training:
            action += self._config.action_noise_range * torch.randn(self._env.action_space.shape)

        action = action.data.numpy()

        if self._action_modifier:
            action = self._action_modifier(observation, action, c)

        return action

    def _get_q(self, batch):
        return self._critic(self._as_tensor(batch["observation"]))

    def _get_target(self, batch):
        # For each observation in batch:
        # target = r + discount_factor * (1 - done) * max_a Q_tar(s, a)
        # a => actions of actor on current observations
        # max_a Q_tar(s, a) = output of critic
        
        start_time = time.time()
        observation_next = self._as_tensor(batch["observation_next"])
        reward = self._as_tensor(batch["reward"]).reshape(-1, 1)
        done = self._as_tensor(batch["done"]).reshape(-1, 1)
        self._tensor_convert_time += time.time() - start_time

        
        start_time = time.time()
        action = self._target_actor(observation_next).reshape(-1, *self._env.action_space.shape)
        q = self._target_critic(observation_next, action)
        self._target_compute_time += time.time() - start_time


        return reward + self._config.discount_factor * (1 - done) * q

    def _flatten_dict(self, inp):
        if type(inp) == dict:
            agent_position = inp["agent_position"]
            target_position = inp["target_position"]
            inp = np.concatenate([agent_position, target_position])
        return inp

    def _update_targets(self, target, main):
        for target_param, main_param in zip(target.parameters(), main.parameters()):
            target_param.data.copy_(self._config.polyak * target_param.data + \
                                    (1 - self._config.polyak) * main_param.data)

    def _update_batch(self):
        start_time = time.time()
        batch = self._replay_buffer.sample(self._config.batch_size)
        self._batch_sample_time += time.time() - start_time
        # Only pick steps in which action was non-zero
        # When a constraint is violated, the safety layer makes action 0 in
        # direction of violating constraint
        # valid_action_mask = np.sum(batch["action"], axis=1) > 0
        # batch = {k: v[valid_action_mask] for k, v in batch.items()}

        # Update critic
        self._critic_optimizer.zero_grad()
        q_target = self._get_target(batch)

        start_time = time.time()
        obs = self._as_tensor(batch["observation"])
        action = self._as_tensor(batch["action"])
        self._tensor_convert_time += time.time() - start_time
        # critic_loss = torch.mean((q_predicted.detach() - q_target) ** 2)
        # Seems to work better


        start_time = time.time()
        q_predicted = self._critic(obs, action)
        critic_loss = F.smooth_l1_loss(q_predicted, q_target)
        critic_loss.backward()
        self._critic_optimizer.step()
        self._critic_update_time += time.time() - start_time

        # Update actor
        self._actor_optimizer.zero_grad()
        # Find loss with updated critic

        start_time = time.time()
        new_action = self._actor(self._as_tensor(batch["observation"])).reshape(-1, *self._env.action_space.shape)
        actor_loss = -torch.mean(self._critic(self._as_tensor(batch["observation"]), new_action))
        actor_loss.backward()
        self._actor_optimizer.step()
        self._actor_update_time += time.time() - start_time

        # Update targets networks
        start_time = time.time()
        self._update_targets(self._target_actor, self._actor)
        self._update_targets(self._target_critic, self._critic)
        self._target_copy_time += time.time() - start_time

        # Log to tensorboard
        start_time = time.time()
        self._writer.add_scalar("critic loss", critic_loss.item(), self._train_global_step)
        self._writer.add_scalar("actor loss", actor_loss.item(), self._train_global_step)
        (seq(self._models.items())
                    .flat_map(lambda x: [(x[0], y) for y in x[1].named_parameters()]) # (model_name, (param_name, param_data))
                    .map(lambda x: (f"{x[0]}_{x[1][0]}", x[1][1]))
                    .for_each(lambda x: self._writer.add_histogram(x[0], x[1].data.numpy(), self._train_global_step)))
        self._train_global_step += 1
        self._logging_time += time.time() - start_time

    def _update(self, episode_length):
        # Update model #episode_length times
        for_each(lambda x: self._update_batch(),
                 range(min(episode_length, self._config.max_updates_per_episode)))

    def evaluate(self, render):
        episode_rewards = []
        episode_lengths = []
        episode_actions = []

        observation = self._env.reset()
        c = self._env.get_constraint_values()
        episode_reward = 0
        episode_length = 0
        episode_action = 0

        self._eval_mode()

        for step in range(self._config.evaluation_steps):
            action = self._get_action(observation, c, is_training=False)
            episode_action += np.absolute(action)
            observation, reward, done, _ = self._env.step(action)
            
            if render:
                self._env.render_env()

            c = self._env.get_constraint_values()
            episode_reward += reward
            episode_length += 1
            
            if done or (episode_length == self._config.max_episode_length):
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_actions.append(episode_action / episode_length)

                observation = self._env.reset()
                c = self._env.get_constraint_values()
                episode_reward = 0
                episode_length = 0
                episode_action = 0

        mean_episode_reward = np.mean(episode_rewards)
        mean_episode_length = np.mean(episode_lengths)

        self._writer.add_scalar("eval mean episode reward", mean_episode_reward, self._eval_global_step)
        self._writer.add_scalar("eval mean episode length", mean_episode_length, self._eval_global_step)
        self._eval_global_step += 1

        self._train_mode()

        print("Validation completed:\n"
              f"Number of episodes: {len(episode_actions)}\n"
              f"Average episode length: {mean_episode_length}\n"
              f"Average reward: {mean_episode_reward}\n"
              f"Average action magnitude: {np.mean(episode_actions)}")

    def train(self, output_folder: str):
        
        start_time = time.time()

        print("==========================================================")
        print("Initializing DDPG training...")
        print("----------------------------------------------------------")
        print(f"Start time: {datetime.fromtimestamp(start_time)}")
        print("==========================================================")

        observation = self._env.reset()
        c = self._env.get_constraint_values()
        episode_reward = 0
        episode_length = 0
        step_trained_on = 0

        number_of_steps = self._config.steps_per_epoch * self._config.epochs

        time_simulating = 0
        time_training = 0
        time_eval = 0

        safety_layer_print = True

        for step in range(number_of_steps):
            sim_start = time.time()
            # Randomly sample episode_ for some initial steps
            
            if step < self._config.start_steps:
                action = self._env.action_space.sample()
            else:
                if safety_layer_print:
                    print("Safety layer is now on")
                    safety_layer_print = False
                action = self._get_action(observation, c)
            
            observation_next, reward, done, _ = self._env.step(action)

            if self._render_training:
                self._env.render_env()
                
            episode_reward += reward
            episode_length += 1

            self._replay_buffer.add({
                "observation": self._flatten_dict(observation),
                "action": action,
                "reward": np.asarray(reward) * self._config.reward_scale,
                "observation_next": self._flatten_dict(observation_next),
                "done": np.asarray(done),
            })

            observation = observation_next
            c = self._env.get_constraint_values()
            sim_end = time.time()
            time_simulating += sim_end - sim_start

            # Make all updates at the end of the episode
            if done or (episode_length == self._config.max_episode_length):
                if step >= self._config.min_buffer_fill and (step - step_trained_on) >= self._config.max_episode_length:
                    update_start = time.time()
                    self._update(episode_length)
                    step_trained_on = step
                    update_end = time.time()
                    time_training += update_end - update_start
                # Reset episode
                observation = self._env.reset()
                c = self._env.get_constraint_values()
                episode_reward = 0
                episode_length = 0
                self._writer.add_scalar("episode length", episode_length)
                self._writer.add_scalar("episode reward", episode_reward)

            # Check if the epoch is over
            # if step != 0 and step % self._config.steps_per_epoch == 0: 
            #     eval_start = time.time()
            #     epoch_number = int(step / self._config.steps_per_epoch)
            #     print(f"Finished epoch {epoch_number}. Running validation ...")
            #     should_render = epoch_number % 10 == 0
            #     self.evaluate(should_render)
            #     eval_end = time.time()
            #     time_eval += eval_end - eval_start
            #     print(f"Simulating: {time_simulating:.2}, Training: {time_training:.2}, Eval: {time_eval:.2}")

            #     print(f"batsh sample: {self._batch_sample_time * 1000:.2}")
            #     print(f"tensor convert: {self._tensor_convert_time * 1000:.2}")
            #     print(f"actor update: {self._actor_update_time * 1000:.2}")
            #     print(f"critic update: {self._critic_update_time * 1000:.2}")
            #     print(f"logging: {self._logging_time * 1000:.2}")
            #     print(f"target copy: {self._target_compute_time * 1000:.2}")
            #     print(f"target compute: {self._target_copy_time * 1000:.2}")
            #     print("----------------------------------------------------------")
            
        
        self._writer.close()
        print("==========================================================")
        print(f"Finished DDPG training. Time spent: {(time.time() - start_time) // 1} secs")
        print("==========================================================")

        self._actor.save(output_folder)
        self._critic.save(output_folder)

