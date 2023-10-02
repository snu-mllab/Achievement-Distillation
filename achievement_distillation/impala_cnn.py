from copy import deepcopy
import math
from typing import Dict, Optional, Sequence, Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from achievement_distillation.torch_util import FanInInitReLULayer


class CnnBasicBlock(nn.Module):
    def __init__(
        self,
        inchan: int,
        init_scale: float = 1.0,
        init_norm_kwargs: Dict = {},
    ):
        super().__init__()

        # Layers
        s = math.sqrt(init_scale)
        self.conv0 = FanInInitReLULayer(
            inchan,
            inchan,
            kernel_size=3,
            padding=1,
            init_scale=s,
            **init_norm_kwargs,
        )
        self.conv1 = FanInInitReLULayer(
            inchan,
            inchan,
            kernel_size=3,
            padding=1,
            init_scale=s,
            **init_norm_kwargs,
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = x + self.conv1(self.conv0(x))
        return x


class CnnDownStack(nn.Module):
    def __init__(
        self,
        inchan: int,
        nblock: int,
        outchan: int,
        init_scale: float = 1.0,
        pool: bool = True,
        post_pool_groups: Optional[int] = None,
        init_norm_kwargs: Dict = {},
        first_conv_norm: bool = False,
        **kwargs,
    ):
        super().__init__()

        # Params
        self.inchan = inchan
        self.outchan = outchan
        self.pool = pool

        # Layers
        first_conv_init_kwargs = deepcopy(init_norm_kwargs)
        if not first_conv_norm:
            first_conv_init_kwargs["group_norm_groups"] = None
            first_conv_init_kwargs["batch_norm"] = False
        self.firstconv = FanInInitReLULayer(
            inchan,
            outchan,
            kernel_size=3,
            padding=1,
            **first_conv_init_kwargs,
        )
        self.post_pool_groups = post_pool_groups
        if post_pool_groups is not None:
            self.n = nn.GroupNorm(post_pool_groups, outchan)
        self.blocks = nn.ModuleList(
            [
                CnnBasicBlock(
                    outchan,
                    init_scale=init_scale / math.sqrt(nblock),
                    init_norm_kwargs=init_norm_kwargs,
                    **kwargs,
                )
                for _ in range(nblock)
            ]
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.firstconv(x)
        if self.pool:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            if self.post_pool_groups is not None:
                x = self.n(x)
        for block in self.blocks:
            x = block(x)
        return x

    def output_shape(self, inshape: Sequence[int]) -> Tuple[int, int, int]:
        c, h, w = inshape
        assert c == self.inchan
        if self.pool:
            return (self.outchan, (h + 1) // 2, (w + 1) // 2)
        else:
            return (self.outchan, h, w)


class ImpalaCNN(nn.Module):
    def __init__(
        self,
        inshape: Sequence[int],
        chans: Sequence[int],
        outsize: int,
        nblock: int,
        init_norm_kwargs: Dict = {},
        dense_init_norm_kwargs: Dict = {},
        first_conv_norm: bool = False,
        **kwargs,
    ):
        super().__init__()

        # Layers
        curshape = inshape
        self.stacks = nn.ModuleList()
        for i, outchan in enumerate(chans):
            stack = CnnDownStack(
                curshape[0],
                nblock=nblock,
                outchan=outchan,
                init_scale=1 / math.sqrt(len(chans)),
                init_norm_kwargs=init_norm_kwargs,
                first_conv_norm=first_conv_norm if i == 0 else True,
                **kwargs,
            )
            self.stacks.append(stack)
            curshape = stack.output_shape(curshape)
        self.dense = FanInInitReLULayer(
            math.prod(curshape),
            outsize,
            layer_type="linear",
            init_scale=1.4,
            **dense_init_norm_kwargs,
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        for stack in self.stacks:
            x = stack(x)
        x = x.reshape(x.size(0), -1)
        x = self.dense(x)
        return x
