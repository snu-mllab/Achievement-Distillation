from typing import Dict

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from achievement_distillation.normalize import NormalizeEwma


class ScaledMSEHead(nn.Module):
    def __init__(
        self,
        insize: int,
        outsize: int,
        init_scale: float = 0.1,
        norm_kwargs: Dict = {},
    ):
        super().__init__()

        # Layer
        self.linear = nn.Linear(insize, outsize)

        # Initialization
        init.orthogonal_(self.linear.weight, gain=init_scale)
        init.constant_(self.linear.bias, val=0.0)

        # Normalizer
        self.normalizer = NormalizeEwma(outsize, **norm_kwargs)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.linear(x)

    def normalize(self, x: th.Tensor) -> th.Tensor:
        return self.normalizer(x)

    def denormalize(self, x: th.Tensor) -> th.Tensor:
        return self.normalizer.denormalize(x)

    def mse_loss(self, pred: th.Tensor, targ: th.Tensor) -> th.Tensor:
        return F.mse_loss(pred, self.normalizer(targ), reduction="none")
