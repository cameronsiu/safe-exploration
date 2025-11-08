from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sim.spawners.from_files import UsdFileCfg

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.assets import AssetBaseCfg

from .turtlebot import TURTLEBOT_CONFIG


@configclass
class ObstacleAvoidSceneCfg(InteractiveSceneCfg):
    
    scene: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Environment",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)
        ),
        spawn=UsdFileCfg(usd_path="./environment_without_people.usd"),
    )

@configclass
class ObstacleAvoidEnvCfg(DirectRLEnvCfg):

    # What is decimation?
    decimation = 2
    episode_length_s = 5.0

    # https://isaac-sim.github.io/IsaacLab/main/source/setup/walkthrough/technical_env_design.html
    # Action, observations, and states spaces can be defined as gymnasium space maybe try gym
    action_space = 2
    observation_space = 3
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=2)

    robot_cfg: ArticulationCfg = TURTLEBOT_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")

    scene: InteractiveSceneCfg = ObstacleAvoidSceneCfg(num_envs=1, env_spacing=40.0)
    
    # Not sure if correct with turtlebot
    dof_names = ["left_wheel_joint", "right_wheel_joint"]
