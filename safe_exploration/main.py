import argparse

from functional import seq
import numpy as np
import torch

import sys

from safe_exploration.core.config import Config
from safe_exploration.env.ballnd import BallND
from safe_exploration.env.spaceship import Spaceship
from safe_exploration.ddpg.actor import Actor
from safe_exploration.ddpg.critic import Critic
from safe_exploration.ddpg.ddpg import DDPG
from safe_exploration.safety_layer.safety_layer import SafetyLayer

def load_args(argv):
    parser = argparse.ArgumentParser(description="Train or Test")

    parser.add_argument("--mode", dest="mode", type=str, default="train")
    parser.add_argument("--out_model_file", dest="out_model_file", type=str, default="output/model.pt")
    parser.add_argument("--model_file", dest="model_file", type=str, default="")

    args = parser.parse_args()

    return args

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def print_ascii_art():
    print(
    """
        _________       _____        ___________              .__                              
        /   _____/____ _/ ____\____   \_   _____/__  _________ |  |   ___________   ___________ 
        \_____  \\__  \\   __\/ __ \   |    __)_\  \/  /\____ \|  |  /  _ \_  __ \_/ __ \_  __ \\
        /        \/ __ \|  | \  ___/   |        \>    < |  |_> >  |_(  <_> )  | \/\  ___/|  | \/
    /_______  (____  /__|  \___  > /_______  /__/\_ \|   __/|____/\____/|__|    \___  >__|   
            \/     \/          \/          \/      \/|__|                           \/    
    """)                                                                                                                  

def train(config):
    ddpg.train()

def test():
    ddpg.test()

def load(self, in_file):
    state_dict = torch.load(in_file)
    self.load_state_dict(state_dict)

def run(args):
    mode = args.mode
    model_file = args.model_file

    # Bottom of defaults.yml
    # TODO: Change this?
    config = Config.get().main.trainer
    # Default seed to 0
    set_seeds(config.seed)

    print_ascii_art()
    print("============================================================")
    print("Initialized SafeExplorer with config:")
    print("------------------------------------------------------------")
    Config.get().pprint()
    print("============================================================")

    env = BallND() if config.task == "ballnd" else Spaceship()
    if config.use_safety_layer:
        safety_layer = SafetyLayer(env)
        safety_layer.train()
    
    observation_dim = (seq(env.observation_space.spaces.values())
                        .map(lambda x: x.shape[0])
                        .sum())

    actor = Actor(observation_dim, env.action_space.shape[0])
    critic = Critic(observation_dim, env.action_space.shape[0])

    safe_action_func = safety_layer.get_safe_action if safety_layer else None
    ddpg = DDPG(env, actor, critic, safe_action_func)
    

    if (model_file != ""):
        ddpg.load(model_file)

    if mode == "train":
        train(ddpg, config)
    elif mode == "test":
        test(ddpg, config)
    
    train(config)

def main(argv):
    args = load_args(argv)
    run(args)

if __name__ == "__main__":
    main(sys.argv)