from functional import seq
from pathlib import Path
import glob

import numpy as np
import torch

from safe_exploration.core.config import Config
from safe_exploration.env.ballnd import BallND
from safe_exploration.env.spaceship import Spaceship
from safe_exploration.env.obstacle_avoid import ObstacleAvoid
from safe_exploration.ddpg.actor import Actor
from safe_exploration.ddpg.critic import Critic
from safe_exploration.ddpg.ddpg import DDPG
from safe_exploration.safety_layer.safety_layer import SafetyLayer


class Trainer:
    def __init__(self):
        self._config = Config.get().main.trainer
        self._set_seeds()

    def _set_seeds(self):
        torch.manual_seed(self._config.seed)
        np.random.seed(self._config.seed)

    def _print_ascii_art(self):
        print(
        """
          _________       _____        ___________              .__                              
         /   _____/____ _/ ____\____   \_   _____/__  _________ |  |   ___________   ___________ 
         \_____  \\__  \\   __\/ __ \   |    __)_\  \/  /\____ \|  |  /  _ \_  __ \_/ __ \_  __ \\
         /        \/ __ \|  | \  ___/   |        \>    < |  |_> >  |_(  <_> )  | \/\  ___/|  | \/
        /_______  (____  /__|  \___  > /_______  /__/\_ \|   __/|____/\____/|__|    \___  >__|   
                \/     \/          \/          \/      \/|__|                           \/    
        """)                                                                                                                  

    def train(self):
        self._print_ascii_art()
        print("============================================================")
        print("Initialized SafeExplorer with config:")
        print("------------------------------------------------------------")
        Config.get().pprint()
        print("============================================================")

        env = BallND() if self._config.task == "ballnd" else \
            ObstacleAvoid() if self._config.task == "obstacleavoid" else \
            Spaceship()

        actor_file = Path(self._config.actor_model_file)
        critic_file = Path(self._config.critic_model_file)

        actor_model_file = None
        if actor_file.exists():
            print(f"Loading actor file {self._config.actor_model_file}")
            actor_model_file = self._config.actor_model_file
        else:
            print(f"Actor model file does not exist {self._config.actor_model_file}")

        critic_model_file = None
        if critic_file.exists():
            print(f"Loading critic file {self._config.critic_model_file}")
            critic_model_file = self._config.critic_model_file
        else:
            print(f"Critic model file does not exist {self._config.critic_model_file}")

        constraint_model_files = glob.glob(self._config.constraint_model_files)
        print(f"Loading constraint model files: {constraint_model_files}")

        safety_layer = None
        if self._config.use_safety_layer:
            safety_layer = SafetyLayer(env, constraint_model_files, render=False)
            
            if not self._config.test:
                safety_layer.train(self._config.output_folder)
            else:
                safety_layer.evaluate()
        else:
            safety_layer = None
        
        observation_dim = (seq(env.observation_space.spaces.values())
                            .map(lambda x: x.shape[0])
                            .sum())

        actor = Actor(observation_dim, env.action_space.shape[0], actor_model_file)
        critic = Critic(observation_dim, env.action_space.shape[0], critic_model_file)

        safe_action_func = safety_layer.get_safe_action if safety_layer else None
        ddpg = DDPG(env, actor, critic, safe_action_func, render_training=False, render_evaluation=True)
        
        if not self._config.test:
            ddpg.train(self._config.output_folder)
        else:
            ddpg.evaluate(False)


if __name__ == '__main__':
    Trainer().train()