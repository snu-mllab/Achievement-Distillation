from __future__ import annotations

from typing import Dict

import torch as th
import torch.nn.functional as F

from gym import spaces

from achievement_distillation.action_head import CategoricalActionHead
from achievement_distillation.mlp import MLP
from achievement_distillation.model.ppo import PPOModel
from achievement_distillation.mse_head import ScaledMSEHead


class PPOADModel(PPOModel):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        hidsize: int,
        impala_kwargs: Dict = {},
        dense_init_norm_kwargs: Dict = {},
        action_head_kwargs: Dict = {},
        mse_head_kwargs: Dict = {},
        nhidlayer: int = 1,
        temperature: float = 0.1,
        use_memory: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            hidsize,
            impala_kwargs=impala_kwargs,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
        )

        # Heads
        num_actions = getattr(self.action_space, "n")
        pi_latent_size = 2 * hidsize if use_memory else hidsize
        vf_latent_size = 2 * hidsize if use_memory else hidsize
        self.pi_head = CategoricalActionHead(
            insize=pi_latent_size,
            num_actions=num_actions,
            **action_head_kwargs,
        )
        self.vf_head = ScaledMSEHead(
            insize=vf_latent_size,
            outsize=1,
            **mse_head_kwargs,
        )

        # AD params
        self.use_memory = use_memory
        self.temperature = temperature

        # AD layers
        num_actions = getattr(self.action_space, "n")
        self.action_mlp = MLP(
            insize=num_actions,
            nhidlayer=nhidlayer,
            outsize=2 * hidsize,
            hidsize=hidsize,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
        )
        self.next_goal_pred_mlp = MLP(
            insize=2 * hidsize,
            nhidlayer=nhidlayer,
            outsize=hidsize,
            hidsize=hidsize,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
        )

    def forward(self, obs: th.Tensor, states: th.Tensor) -> Dict[str, th.Tensor]:
        # Pass through encoder
        latents = self.encode(obs)

        # Concatenate latents and states
        if self.use_memory:
            pi_latents = vf_latents = th.cat([latents, states], dim=-1)
        else:
            pi_latents = vf_latents = latents

        # Pass through heads
        pi_logits = self.pi_head(pi_latents)
        vpreds = self.vf_head(vf_latents)

        # Define outputs
        outputs = {
            "latents": latents,
            "pi_latents": pi_latents,
            "vf_latents": vf_latents,
            "pi_logits": pi_logits,
            "vpreds": vpreds,
        }

        return outputs

    def film(self, latents: th.Tensor, actions: th.Tensor) -> th.Tensor:
        # Change actions to one-hot vectors
        num_actions = getattr(self.action_space, "n")
        onehot_actions = th.eye(num_actions).to(actions.device)[actions.squeeze(dim=-1)]

        # Get action latents
        action_latents = self.action_mlp.forward(onehot_actions)

        # Combine obs and action latents
        gamma, beta = th.chunk(action_latents, 2, dim=-1)
        latents = (1 + gamma) * latents + beta

        return latents

    def get_states(self, goal_obs: th.Tensor, goal_next_obs: th.Tensor) -> th.Tensor:
        # Get zero obs and zero next obs
        batch_size = goal_obs.shape[0]
        zero_obs_conds = goal_obs.reshape(batch_size, -1) == 0
        zero_obs_conds = zero_obs_conds.all(dim=-1, keepdim=True)
        zero_next_obs_conds = goal_next_obs.reshape(batch_size, -1) == 0
        zero_next_obs_conds = zero_next_obs_conds.all(dim=-1, keepdim=True)

        # Get goal latents and goal next latents
        goal_latents = self.encode(goal_obs)
        goal_next_latents = self.encode(goal_next_obs)

        # Mask out latents to zeros
        goal_latents = th.where(zero_obs_conds, 0, goal_latents)
        goal_next_latents = th.where(zero_next_obs_conds, 0, goal_next_latents)

        # Get states
        states = goal_next_latents - goal_latents
        states = F.normalize(states, dim=-1)

        return states

    def get_next_goal_preds(
        self,
        latents: th.Tensor,
        actions: th.Tensor,
        states: th.Tensor,
    ) -> th.Tensor:
        # Pass through FiLM
        latents = self.film(latents, actions)

        # Concatenate latents and states
        if self.use_memory:
            latents = th.cat([latents, states], dim=-1)

        # Pass through MLP
        next_goal_preds = self.next_goal_pred_mlp(latents)

        # Normalize preds
        next_goal_preds = F.normalize(next_goal_preds, dim=-1)

        return next_goal_preds

    def compute_pred_losses(
        self,
        anc_goal_obs: th.Tensor,
        anc_goal_next_obs: th.Tensor,
        pos_obs: th.Tensor,
        pos_actions: th.Tensor,
        pos_old_states: th.Tensor,
        pos_old_vtargs: th.Tensor,
        neg_obs: th.Tensor,
        neg_actions: th.Tensor,
        neg_old_states: th.Tensor,
        neg_old_vtargs: th.Tensor,
        old_model: PPOADModel,
    ) -> Dict[str, th.Tensor]:
        # Process anchor
        with th.no_grad():
            anc_states = self.get_states(anc_goal_obs, anc_goal_next_obs)

        # Process positive
        pos_outputs = self.forward(pos_obs, states=pos_old_states)
        pos_old_outputs = old_model.act(pos_obs, states=pos_old_states)
        pos_latents = pos_outputs["latents"]
        pos_preds = self.get_next_goal_preds(pos_latents, pos_actions, pos_old_states)

        # Process negative
        neg_outputs = self.forward(neg_obs, states=neg_old_states)
        neg_old_outputs = old_model.act(neg_obs, states=neg_old_states)
        neg_latents = neg_outputs["latents"]
        neg_preds = self.get_next_goal_preds(neg_latents, neg_actions, neg_old_states)

        # Compute pred loss
        pos_logits = th.einsum("bk,bk->b", anc_states, pos_preds)
        neg_logits = th.einsum("bk,bk->b", anc_states, neg_preds)
        logits = th.stack([pos_logits, neg_logits], dim=-1)
        logits = logits / self.temperature
        targets = th.zeros(len(logits)).to(logits.device).long()
        pred_loss = F.cross_entropy(logits, targets)

        # Compute policy dist
        pos_pi_logits = pos_outputs["pi_logits"]
        pos_old_pi_logits = pos_old_outputs["pi_logits"]
        neg_pi_logits = neg_outputs["pi_logits"]
        neg_old_pi_logits = neg_old_outputs["pi_logits"]
        pi_logits = th.cat([pos_pi_logits, neg_pi_logits], dim=0)
        old_pi_logits = th.cat([pos_old_pi_logits, neg_old_pi_logits], dim=0)
        pi_dist = self.pi_head.kl_divergence(pi_logits, old_pi_logits).mean()

        # Compute value dist
        pos_vpreds = pos_outputs["vpreds"]
        neg_vpreds = neg_outputs["vpreds"]
        vpreds = th.cat([pos_vpreds, neg_vpreds], dim=0)
        old_vtargs = th.cat([pos_old_vtargs, neg_old_vtargs], dim=0)
        vf_dist = self.vf_head.mse_loss(vpreds, old_vtargs).mean()

        # Define pred losses
        pred_losses = {
            "pred_loss": pred_loss,
            "pi_dist": pi_dist,
            "vf_dist": vf_dist,
        }

        return pred_losses

    def compute_match_losses(
        self,
        anc_goal_obs: th.Tensor,
        anc_goal_next_obs: th.Tensor,
        pos_goal_obs: th.Tensor,
        pos_goal_next_obs: th.Tensor,
        neg_goal_obs: th.Tensor,
        neg_goal_next_obs: th.Tensor,
        obs: th.Tensor,
        old_states: th.Tensor,
        old_vtargs: th.Tensor,
        old_model: PPOADModel,
    ) -> Dict[str, th.Tensor]:
        # Process anchor
        anc_states = self.get_states(anc_goal_obs, anc_goal_next_obs)

        # Process positive
        with th.no_grad():
            pos_states = self.get_states(pos_goal_obs, pos_goal_next_obs)

        # Process negative
        with th.no_grad():
            neg_states = self.get_states(neg_goal_obs, neg_goal_next_obs)

        # Process misc
        outputs = self.forward(obs, states=old_states)
        old_outputs = old_model.act(obs, states=old_states)

        # Compute match loss
        pos_logits = th.einsum("bk,bk->b", anc_states, pos_states)
        neg_logits = th.einsum("bk,bk->b", anc_states, neg_states)
        logits = th.stack([pos_logits, neg_logits], dim=-1)
        logits = logits / self.temperature
        targets = th.zeros(len(logits)).to(logits.device).long()
        match_loss = F.cross_entropy(logits, targets)

        # Compute policy dist
        pi_logits = outputs["pi_logits"]
        old_pi_logits = old_outputs["pi_logits"]
        pi_dist = self.pi_head.kl_divergence(pi_logits, old_pi_logits).mean()

        # Compute value dist
        vpreds = outputs["vpreds"]
        vf_dist = self.vf_head.mse_loss(vpreds, old_vtargs).mean()

        # Define match losses
        match_losses = {
            "match_loss": match_loss,
            "pi_dist": pi_dist,
            "vf_dist": vf_dist,
        }

        return match_losses
