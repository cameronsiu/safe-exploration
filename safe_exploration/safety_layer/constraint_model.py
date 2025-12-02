import torch
from torch.nn.init import uniform_

from safe_exploration.core.config import Config
from safe_exploration.core.net import Net


class ConstraintModel(Net):
    def __init__(self, observation_dim, action_dim, model_file:str=None):
        config = Config.get().safety_layer.constraint_model
        
        super(ConstraintModel, self)\
            .__init__(observation_dim,
                      action_dim,
                      config.layers,
                      config.init_bound,
                      uniform_,
                      None)

        if model_file is not None:
            state_dict = torch.load(model_file)
            self.load_state_dict(state_dict)

    def save(self, output_folder:str, model_index:int):
        state_dict = self.state_dict()
        torch.save(state_dict, f"{output_folder}/constraint_{model_index}.pt")

