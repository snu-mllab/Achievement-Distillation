import abc
from typing import Dict

import torch as th

from achievement_distillation.model.base import BaseModel
from achievement_distillation.storage import RolloutStorage


class BaseAlgorithm(abc.ABC):
    def __init__(self, model: BaseModel):
        self.model = model

    @abc.abstractclassmethod
    def update(self, storage: RolloutStorage) -> Dict[str, th.Tensor]:
        pass