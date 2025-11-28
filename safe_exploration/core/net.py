from functional import seq
from torch.nn import Linear, Module, ModuleList, LayerNorm 
from torch.nn.init import uniform_
import torch.nn.functional as F


class Net(Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 layer_dims,
                 init_bound,
                 initializer,
                 last_activation,
                 action_scale=None):
        super(Net, self).__init__()

        self._initializer = initializer
        self._last_activation = last_activation
        self._action_scale = action_scale

        _layer_dims = [in_dim] + layer_dims + [out_dim]
        self._layers = ModuleList(seq(_layer_dims[:-1])
                                    .zip(_layer_dims[1:])
                                    .map(lambda x: Linear(x[0], x[1]))
                                    .to_list())
        self.norm = LayerNorm(layer_dims[0])

        self._init_weights(init_bound)
    
    def _init_weights(self, bound):
        # Initialize all layers except the last one with fan-in initializer
        (seq(self._layers[:-1])
            .map(lambda x: x.weight)
            .for_each(self._initializer))
        # Init last layer with uniform initializer
        uniform_(self._layers[-1].weight, -bound, bound)

    def forward(self, inp):
        out = inp

        for i, layer in enumerate(self._layers[:-1]):
            if i == 0:
                out = F.relu(self.norm(layer(out)))
            else:
                out = F.relu(layer(out))

        if self._last_activation:
            out = self._last_activation(self._layers[-1](out))
        else:
            out = self._layers[-1](out)

        if self._action_scale is not None:
            out = self._action_scale * out

        return out