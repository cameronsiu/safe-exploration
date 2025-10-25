import torch
from torch.nn import Linear, Module
import torch.nn.functional as F

from safe_exploration.core.config import Config
from safe_exploration.core.net import Net
from safe_exploration.ddpg.utils import init_fan_in_uniform

class Critic(Module):
    def __init__(self, observation_dim, action_dim, model_file:str=None):
        super(Critic, self).__init__()
        
        config = Config.get().ddpg.critic

        self._observation_linear = Linear(observation_dim, config.layers[0])
        self._action_linear = Linear(action_dim, config.layers[0])
        
        init_fan_in_uniform(self._observation_linear.weight)
        init_fan_in_uniform(self._action_linear.weight)

        self._model = Net(config.layers[0] * 2,
                          1,
                          config.layers[1:],
                          config.init_bound,
                          init_fan_in_uniform,
                          None)

        if model_file is not None:
            state_dict = torch.load(model_file)
            self.load_state_dict(state_dict)

    def forward(self, observation, action):
        observation_ = F.relu(self._observation_linear(observation))
        action_ = F.relu(self._action_linear(action))
        return self._model(torch.cat([observation_, action_], dim=1))
    
    def save(self, output_folder:str):
        state_dict = self.state_dict()
        torch.save(state_dict, f"{output_folder}/critic.pt")

