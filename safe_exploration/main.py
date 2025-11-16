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

        if self._config.task == "obstacleavoidisaaclab":
            env = self.isaaclab_env()
        else:
            env = ObstacleAvoid()

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
            ddpg.evaluate()

    def isaaclab_env(self):
        import argparse
        from isaaclab.app import AppLauncher
        from isaacsim.simulation_app import SimulationApp

        args = argparse.Namespace(
            num_envs=self._config.num_envs,
            device=self._config.device,
            headless=not self._config.render_training,
        )

        app_launcher = AppLauncher(args)
        sim_app: SimulationApp = app_launcher.app

        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.assets.articulation import ArticulationCfg


        # HACK https://github.com/isaac-sim/IsaacLab/discussions/2256
        # Setting enable_scene_query_support to true because lidar values were not being updated
        # Also, add  "isaacsim.sensors.physx" = {}  inside of dependencies isaaclab.python.headless.kit 
        sim_cfg = sim_utils.SimulationCfg(device=self._config.device, enable_scene_query_support=True)
        sim_context = sim_utils.SimulationContext(sim_cfg)
        sim_context.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

        TURTLEBOT_CONFIG = ArticulationCfg(
            spawn=sim_utils.UsdFileCfg(usd_path=f"safe_exploration/env/turtlebot.usd"),
            actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
        )

        class ObstacleAvoidCfg(InteractiveSceneCfg):
            """Obstacle Avoid Scene."""

            scene: AssetBaseCfg = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Environment",
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)
                ),
                spawn=sim_utils.UsdFileCfg(usd_path="safe_exploration/env/obstacle_avoid.usd"),
                debug_vis=True,
            )

            Turtlebot: ArticulationCfg = TURTLEBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Turtlebot")

        scene_cfg = ObstacleAvoidCfg(self._config.num_envs, env_spacing=2.0)
        scene = InteractiveScene(scene_cfg)

        env = ObstacleAvoidIsaacLab(sim_app, sim_context, scene)
        return env


if __name__ == '__main__':
    Trainer().train()