import torch

from safe_exploration.core.config import Config
from safe_exploration.core.net import Net
from safe_exploration.ddpg.utils import init_fan_in_uniform


class Actor(Net):
    def __init__(self, observation_dim, action_dim):
        config = Config.get().ddpg.actor

        super(Actor, self).__init__(observation_dim,
                                    action_dim,
                                    config.layers,
                                    config.init_bound,
                                    init_fan_in_uniform,
                                    torch.tanh)