from typing import Dict, Iterator

import torch as th
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from gym import spaces

from achievement_distillation.model.base import BaseModel


class RolloutStorage:
    def __init__(
        self,
        nstep: int,
        nproc: int,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        hidsize: int,
        device: th.device,
    ):
        # Params
        self.nstep = nstep
        self.nproc = nproc
        self.device = device

        # Get obs shape and action dim
        assert isinstance(observation_space, spaces.Box)
        assert isinstance(action_space, spaces.Discrete)
        obs_shape = getattr(observation_space, "shape")
        action_shape = (1,)

        # Tensors
        self.obs = th.zeros(nstep + 1, nproc, *obs_shape, device=device)
        self.actions = th.zeros(nstep, nproc, *action_shape, device=device).long()
        self.rewards = th.zeros(nstep, nproc, 1, device=device)
        self.masks = th.ones(nstep + 1, nproc, 1, device=device)
        self.vpreds = th.zeros(nstep + 1, nproc, 1, device=device)
        self.log_probs = th.zeros(nstep, nproc, 1, device=device)
        self.returns = th.zeros(nstep, nproc, 1, device=device)
        self.advs = th.zeros(nstep, nproc, 1, device=device)
        self.successes = th.zeros(nstep + 1, nproc, 22, device=device).long()
        self.timesteps = th.zeros(nstep + 1, nproc, 1, device=device).long()
        self.states = th.zeros(nstep + 1, nproc, hidsize, device=device)

        # Step
        self.step = 0

    def __getitem__(self, key: str) -> th.Tensor:
        return getattr(self, key)

    def get_inputs(self, step: int):
        inputs = {"obs": self.obs[step], "states": self.states[step]}
        return inputs

    def insert(
        self,
        obs: th.Tensor,
        latents: th.Tensor,
        actions: th.Tensor,
        rewards: th.Tensor,
        masks: th.Tensor,
        vpreds: th.Tensor,
        log_probs: th.Tensor,
        successes: th.Tensor,
        model: BaseModel,
        **kwargs,
    ):
        # Get prev successes, timesteps, and states
        prev_successes = self.successes[self.step]
        prev_states = self.states[self.step]
        prev_timesteps = self.timesteps[self.step]

        # Update timesteps
        timesteps = prev_timesteps + 1

        # Update states if succeeded
        success_conds = (successes != prev_successes).any(dim=1, keepdim=True).float()
        if success_conds.any():
            with th.no_grad():
                next_latents = model.encode(obs)
            states = next_latents - latents
            states = F.normalize(states, dim=-1)
            states = (1 - success_conds) * prev_states + success_conds * states
        else:
            states = prev_states

        # Reset successes, timesteps, and states if done
        done_conds = 1 - masks
        successes = (1 - done_conds) * successes + done_conds * th.zeros_like(successes)
        timesteps = (1 - done_conds) * timesteps + done_conds * th.zeros_like(timesteps)
        states = (1 - done_conds) * states + done_conds * th.zeros_like(states)

        # Update tensors
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.vpreds[self.step].copy_(vpreds)
        self.log_probs[self.step].copy_(log_probs)
        self.successes[self.step + 1].copy_(successes)
        self.timesteps[self.step + 1].copy_(timesteps)
        self.states[self.step + 1].copy_(states)

        # Update step
        self.step = (self.step + 1) % self.nstep

    def reset(self):
        # Reset tensors
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.successes[0].copy_(self.successes[-1])
        self.timesteps[0].copy_(self.timesteps[-1])
        self.states[0].copy_(self.states[-1])

        # Reset step
        self.step = 0

    def compute_returns(self, gamma: float, gae_lambda: float):
        # Compute return
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            delta = (
                self.rewards[step]
                + gamma * self.vpreds[step + 1] * self.masks[step + 1]
                - self.vpreds[step]
            )
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.vpreds[step]
            self.advs[step] = gae

        # Compute advantage
        self.advs = (self.advs - self.advs.mean()) / (self.advs.std() + 1e-8)

    def get_data_loader(self, nbatch: int) -> Iterator[Dict[str, th.Tensor]]:
        # Get sampler
        ndata = self.nstep * self.nproc
        assert ndata >= nbatch
        batch_size = ndata // nbatch
        sampler = SubsetRandomSampler(range(ndata))
        sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)

        # Sample batch
        obs = self.obs[:-1].view(-1, *self.obs.shape[2:])
        states = self.states[:-1].view(-1, *self.states.shape[2:])
        actions = self.actions.view(-1, *self.actions.shape[2:])
        vtargs = self.returns.view(-1, *self.returns.shape[2:])
        log_probs = self.log_probs.view(-1, *self.log_probs.shape[2:])
        advs = self.advs.view(-1, *self.advs.shape[2:])

        for indices in sampler:
            batch = {
                "obs": obs[indices],
                "states": states[indices],
                "actions": actions[indices],
                "vtargs": vtargs[indices],
                "log_probs": log_probs[indices],
                "advs": advs[indices],
            }
            yield batch
