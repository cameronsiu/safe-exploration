from functional import seq
from pathlib import Path
import glob

import numpy as np
import torch

from safe_exploration.core.config import Config
from safe_exploration.env.obstacle_avoid import ObstacleAvoid
from safe_exploration.env.obstacle_avoid_isaaclab import ObstacleAvoidIsaacLab
from safe_exploration.ddpg.actor import Actor
from safe_exploration.ddpg.critic import Critic
from safe_exploration.ddpg.ddpg import DDPG
from safe_exploration.safety_layer.safety_layer import SafetyLayer

class MockEnv:
    def get_num_constraints(self):
        return 1

class SafetyLayerTester:
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

    def get_constraint_values(self, lidar_readings):
        clipped_readings = np.clip(lidar_readings, 0, self._config.constraint_max_clip)
        return np.array([self._config.agent_slack - np.min(clipped_readings)])

    def test(self):
        self._print_ascii_art()
        print("============================================================")
        print("Initialized SafeExplorer with config:")
        print("------------------------------------------------------------")
        Config.get().pprint()
        print("============================================================")

        constraint_model_files = glob.glob(self._config.constraint_model_files)
        if len(constraint_model_files) == 0:
            raise Exception(f"Could not find files for pattern: {self._config.constraint_model_files}")
        else:
            print(f"Loading constraint model files: {constraint_model_files}")

        mock_env = MockEnv()
        safety_layer = SafetyLayer(mock_env, constraint_model_files, render=self._config.render_training)



if __name__ == '__main__':
    SafetyLayerTester().test()