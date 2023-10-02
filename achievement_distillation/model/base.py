from __future__ import annotations

import abc
from typing import Dict

import torch as th
import torch.nn as nn

from gym import spaces


class BaseModel(nn.Module, abc.ABC):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
    ):
        super().__init__()

        # Observation and action spaces
        self.observation_space = observation_space
        self.action_space = action_space

    @abc.abstractclassmethod
    def act(self, obs: th.Tensor, **kwargs) -> Dict[str, th.Tensor]:
        pass

    @abc.abstractclassmethod
    def forward(self, obs: th.Tensor) -> Dict[str, th.Tensor]:
        pass

    @abc.abstractclassmethod
    def encode(self, obs: th.Tensor) -> th.Tensor:
        pass

    @abc.abstractclassmethod
    def compute_losses(self, **kwargs) -> Dict[str, th.Tensor]:
        pass
