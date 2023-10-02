from typing import Tuple

import torch as th
import torch.nn as nn


class NormalizeEwma(nn.Module):
    def __init__(
        self,
        insize: int,
        norm_axes: int = 1,
        beta: float = 0.99,
        epsilon: float = 1e-2,
    ):
        super().__init__()

        # Params
        self.norm_axes = norm_axes
        self.beta = beta
        self.epsilon = epsilon

        # Parameters
        self.running_mean = nn.Parameter(th.zeros(insize), requires_grad=False)
        self.running_mean_sq = nn.Parameter(th.zeros(insize), requires_grad=False)
        self.debiasing_term = nn.Parameter(th.tensor(0.0), requires_grad=False)

    def running_mean_var(self) -> Tuple[th.Tensor, th.Tensor]:
        mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        var = (mean_sq - mean**2).clamp(min=1e-2)
        return mean, var

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.training:
            x_detach = x.detach()
            batch_mean = x_detach.mean(dim=tuple(range(self.norm_axes)))
            batch_mean_sq = (x_detach**2).mean(dim=tuple(range(self.norm_axes)))

            weight = self.beta
            self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
            self.running_mean_sq.mul_(weight).add_(batch_mean_sq * (1.0 - weight))
            self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

        mean, var = self.running_mean_var()
        mean = mean[(None,) * self.norm_axes]
        var = var[(None,) * self.norm_axes]
        x = (x - mean) / th.sqrt(var)
        return x

    def denormalize(self, x: th.Tensor) -> th.Tensor:
        mean, var = self.running_mean_var()
        mean = mean[(None,) * self.norm_axes]
        var = var[(None,) * self.norm_axes]
        x = x * th.sqrt(var) + mean
        return x
