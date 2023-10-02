import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class CategoricalActionHead(nn.Module):
    def __init__(
        self,
        insize: int,
        num_actions: int,
        init_scale: float = 0.01,
    ):
        super().__init__()

        # Layer
        self.linear = nn.Linear(insize, num_actions)

        # Initialization
        init.orthogonal_(self.linear.weight, gain=init_scale)
        init.constant_(self.linear.bias, val=0.0)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.linear(x)
        logits = F.log_softmax(x, dim=-1)
        return logits

    def log_prob(self, logits: th.Tensor, actions: th.Tensor) -> th.Tensor:
        log_prob = th.gather(logits, dim=-1, index=actions)
        return log_prob

    def entropy(self, logits: th.Tensor) -> th.Tensor:
        probs = th.exp(logits)
        entropy = -th.sum(probs * logits, dim=-1, keepdim=True)
        return entropy

    def sample(self, logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        if deterministic:
            actions = th.argmax(logits, dim=-1, keepdim=True)
        else:
            u = th.rand_like(logits)
            u[u == 1.0] = 0.999
            gumbels = logits - th.log(-th.log(u))
            actions = th.argmax(gumbels, dim=-1, keepdim=True)
        return actions

    def kl_divergence(self, logits_q: th.Tensor, logits_p: th.Tensor) -> th.Tensor:
        kl = th.sum(th.exp(logits_q) * (logits_q - logits_p), dim=-1, keepdim=True)
        return kl
