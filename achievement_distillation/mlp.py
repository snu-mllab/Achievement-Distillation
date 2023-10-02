from typing import Dict

import torch as th
import torch.nn as nn

from achievement_distillation.torch_util import FanInInitReLULayer


class MLP(nn.Module):
    def __init__(
        self,
        insize: int,
        nhidlayer: int,
        outsize: int,
        hidsize: int,
        dense_init_norm_kwargs: Dict = {},
    ):
        super().__init__()

        # Layers
        insizes = [insize] + nhidlayer * [hidsize]
        outsizes = nhidlayer * [hidsize] + [outsize]
        self.layers = nn.ModuleList()

        for i, (insize, outsize) in enumerate(zip(insizes, outsizes)):
            use_activation = i < nhidlayer
            init_scale = 1.4 if use_activation else 1.0
            init_norm_kwargs = dense_init_norm_kwargs if use_activation else {}
            layer = FanInInitReLULayer(
                insize,
                outsize,
                layer_type="linear",
                use_activation=use_activation,
                init_scale=init_scale,
                **init_norm_kwargs,
            )
            self.layers.append(layer)

    def forward(self, x: th.Tensor) -> th.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
