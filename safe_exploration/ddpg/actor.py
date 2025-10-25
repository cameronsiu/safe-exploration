import torch

from safe_exploration.core.config import Config
from safe_exploration.core.net import Net
from safe_exploration.ddpg.utils import init_fan_in_uniform


class Actor(Net):
    def __init__(self, observation_dim, action_dim, model_file:str=None):
        config = Config.get().ddpg.actor

        super(Actor, self).__init__(observation_dim,
                                    action_dim,
                                    config.layers,
                                    config.init_bound,
                                    init_fan_in_uniform,
                                    torch.tanh)
        
        if model_file is not None:
            state_dict = torch.load(model_file)
            self.load_state_dict(state_dict)
    
    def save(self, output_folder:str):
        state_dict = self.state_dict()
        torch.save(state_dict, f"{output_folder}/actor.pt")

