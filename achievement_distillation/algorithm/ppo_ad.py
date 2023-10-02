from collections import deque
import copy
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch as th
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from ot.partial import entropic_partial_wasserstein

from achievement_distillation.algorithm.base import BaseAlgorithm
from achievement_distillation.model.ppo_ad import PPOADModel
from achievement_distillation.storage import RolloutStorage


class Buffer:
    def __init__(self, maxlen: int):
        self.segs: List[Dict[str, th.Tensor]] = deque(maxlen=maxlen)
        self.trajs: List[Dict[str, th.Tensor]] = []

    def __len__(self):
        return len(self.segs)

    def insert(self, seg: Dict[str, th.Tensor]):
        self.segs.append(seg)

    def parse_segs(self):
        # Clear trajectories
        self.trajs.clear()

        # Concatenate segments
        obs = th.cat([seg["obs"][:-1] for seg in self.segs], dim=0)
        actions = th.cat([seg["actions"] for seg in self.segs], dim=0)
        states = th.cat([seg["states"][:-1] for seg in self.segs], dim=0)
        returns = th.cat([seg["returns"] for seg in self.segs], dim=0)
        masks = th.cat([seg["masks"][:-1] for seg in self.segs], dim=0)
        successes = th.cat([seg["successes"][:-1] for seg in self.segs], dim=0)

        # Sanity check
        assert (
            len(obs)
            == len(actions)
            == len(states)
            == len(returns)
            == len(masks)
            == len(successes)
        )

        # Split into trajectories
        nproc = obs.shape[1]

        for p in range(nproc):
            # Get segment per process
            obs_p = obs[:, p]
            actions_p = actions[:, p]
            states_p = states[:, p]
            returns_p = returns[:, p]
            masks_p = masks[:, p]
            successes_p = successes[:, p]

            # Get done steps
            done_conds_p = (masks_p == 0).squeeze(dim=-1)
            done_steps_p = done_conds_p.nonzero().squeeze(dim=-1)
            done_steps_p = done_steps_p.tolist()
            done_steps_p = sorted(done_steps_p)

            for start, end in zip(done_steps_p[:-1], done_steps_p[1:]):
                # Get trajectory
                obs_t = obs_p[start:end]
                actions_t = actions_p[start:end]
                states_t = states_p[start:end]
                returns_t = returns_p[start:end]
                successes_t = successes_p[start:end]

                # Store trajectory
                traj = {
                    "obs": obs_t,
                    "actions": actions_t,
                    "old_states": states_t,
                    "old_vtargs": returns_t,
                    "successes": successes_t,
                }
                self.trajs.append(traj)

    def preprocess_trajs(self):
        # Loop over trajectories
        for traj in self.trajs:
            # Get obs and successes
            obs = traj["obs"]
            successes = traj["successes"]

            # Get goals
            goals = self.get_goals(obs, successes)

            # Update trajectory
            traj.update(goals)

    def get_goals(
        self,
        obs: th.Tensor,
        successes: th.Tensor,
    ) -> Dict[str, th.Tensor]:
        # Get goal steps
        goal_conds = (successes[1:] != successes[:-1]).any(dim=-1)
        goal_steps = goal_conds.nonzero().squeeze(dim=-1)
        goal_steps = goal_steps + 1

        # Get goal obs and goal next obs
        if len(goal_steps) == 0:
            goal_obs = th.zeros(0, *obs.shape[1:])
            goal_next_obs = th.zeros(0, *obs.shape[1:])
        else:
            goal_obs = obs[goal_steps - 1]
            goal_next_obs = obs[goal_steps]

        # Define goals
        goals = {
            "goal_steps": goal_steps,
            "goal_obs": goal_obs,
            "goal_next_obs": goal_next_obs,
        }

        return goals

    def get_next_goals(
        self,
        goal_steps: th.Tensor,
        goal_obs: th.Tensor,
        goal_next_obs: th.Tensor,
        obs: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        # Get goal steps
        next_goal_obs = []
        next_goal_next_obs = []
        goal_steps = goal_steps.tolist()
        goal_steps = sorted(set([0] + goal_steps + [len(obs)]))

        # Get next goal obs and next goal next obs
        for i, (start, end) in enumerate(zip(goal_steps[:-1], goal_steps[1:])):
            if i == len(goal_steps) - 2:
                # Zero next obs for no next goal
                next_goal_ob = obs[-1].unsqueeze(dim=0)
                next_goal_next_ob = th.zeros_like(obs[-1]).unsqueeze(dim=0)
            else:
                next_goal_ob = goal_obs[i].unsqueeze(dim=0)
                next_goal_next_ob = goal_next_obs[i].unsqueeze(dim=0)

            next_goal_ob = next_goal_ob.repeat_interleave(end - start, dim=0)
            next_goal_obs.append(next_goal_ob)

            next_goal_next_ob = next_goal_next_ob.repeat_interleave(end - start, dim=0)
            next_goal_next_obs.append(next_goal_next_ob)

        next_goal_obs = th.cat(next_goal_obs, dim=0)
        next_goal_next_obs = th.cat(next_goal_next_obs, dim=0)

        return next_goal_obs, next_goal_next_obs

    def get_pred_data_loader(
        self,
        max_batch_size: int = 512,
    ) -> Iterator[Dict[str, th.Tensor]]:
        # Loop over trajectories
        ntraj = len(self.trajs)

        for i in th.randperm(ntraj):
            # Get trajectory
            traj = self.trajs[i]
            obs = traj["obs"]
            actions = traj["actions"]
            old_states = traj["old_states"]
            old_vtargs = traj["old_vtargs"]
            goal_steps = traj["goal_steps"]
            goal_obs = traj["goal_obs"]
            goal_next_obs = traj["goal_next_obs"]

            # Continue if no goal
            if len(goal_steps) == 0:
                continue

            # Get next goals
            next_goal_obs, next_goal_next_obs = self.get_next_goals(
                goal_steps,
                goal_obs,
                goal_next_obs,
                obs,
            )

            # Sanity check
            assert len(obs) == len(next_goal_obs)

            # Get anchor
            anc_goal_obs = next_goal_obs
            anc_goal_next_obs = next_goal_next_obs

            # Get positive
            pos_obs = obs
            pos_actions = actions
            pos_old_states = old_states
            pos_old_vtargs = old_vtargs

            # Get negative
            ndata = len(obs)
            rand_steps = th.randperm(ndata)
            neg_obs = obs[rand_steps]
            neg_actions = actions[rand_steps]
            neg_old_states = old_states[rand_steps]
            neg_old_vtargs = old_vtargs[rand_steps]

            # Get sampler
            sampler = SubsetRandomSampler(range(ndata))
            sampler = BatchSampler(sampler, batch_size=max_batch_size, drop_last=False)

            for inds in sampler:
                batch = {
                    # Anchor
                    "anc_goal_obs": anc_goal_obs[inds].cuda(),
                    "anc_goal_next_obs": anc_goal_next_obs[inds].cuda(),
                    # Positive
                    "pos_obs": pos_obs[inds].cuda(),
                    "pos_actions": pos_actions[inds].cuda(),
                    "pos_old_states": pos_old_states[inds].cuda(),
                    "pos_old_vtargs": pos_old_vtargs[inds].cuda(),
                    # Negative
                    "neg_obs": neg_obs[inds].cuda(),
                    "neg_actions": neg_actions[inds].cuda(),
                    "neg_old_states": neg_old_states[inds].cuda(),
                    "neg_old_vtargs": neg_old_vtargs[inds].cuda(),
                }
                yield batch

    def get_match_data_loader(
        self,
        model: PPOADModel,
        max_batch_size: int = 512,
    ) -> Iterator[Dict[str, th.Tensor]]:
        # Filter trajectories
        trajs = [traj for traj in self.trajs if len(traj["goal_steps"]) > 0]

        # Loop over trajectories
        ntraj = len(trajs)

        for i in th.randperm(ntraj):
            # Get source trajectory
            traj_s = trajs[i]
            obs_s = traj_s["obs"]
            old_states_s = traj_s["old_states"]
            old_vtargs_s = traj_s["old_vtargs"]
            goal_obs_s = traj_s["goal_obs"]
            goal_next_obs_s = traj_s["goal_next_obs"]

            # Compute source states
            with th.no_grad():
                goal_obs_s = goal_obs_s.cuda()
                goal_next_obs_s = goal_next_obs_s.cuda()
                states_s = model.get_states(goal_obs_s, goal_next_obs_s)

            # Sample trajectories
            anc_goal_obs = []
            anc_goal_next_obs = []
            pos_goal_obs = []
            pos_goal_next_obs = []
            neg_goal_obs = []
            neg_goal_next_obs = []

            inds = th.randperm(ntraj - 1)[:16]

            for j in inds:
                if j >= i:
                    j += 1

                # Get target trajectory
                traj_t = trajs[j]
                goal_obs_t = traj_t["goal_obs"]
                goal_next_obs_t = traj_t["goal_next_obs"]

                # Compute target states
                with th.no_grad():
                    goal_obs_t = goal_obs_t.cuda()
                    goal_next_obs_t = goal_next_obs_t.cuda()
                    states_t = model.get_states(goal_obs_t, goal_next_obs_t)

                # Match source and target states
                a = np.ones(len(states_s))
                b = np.ones(len(states_t))
                M = 1 - th.einsum("ik,jk->ij", states_s, states_t).cpu().numpy()
                T = entropic_partial_wasserstein(a, b, M, reg=0.05, numItermax=100)
                T = th.from_numpy(T).float()
                row_inds, col_inds = th.where(T > 0.5)

                # Continue if no matching
                if len(row_inds) == 0:
                    continue

                # Get anchor
                anc_goal_obs.append(goal_obs_s[row_inds])
                anc_goal_next_obs.append(goal_next_obs_s[row_inds])

                # Get positive
                pos_goal_obs.append(goal_obs_t[col_inds])
                pos_goal_next_obs.append(goal_next_obs_t[col_inds])

                # Get negative
                rand_inds = th.randint(len(goal_obs_t), (len(col_inds),))
                neg_goal_obs.append(goal_obs_t[rand_inds])
                neg_goal_next_obs.append(goal_next_obs_t[rand_inds])

            # Continue if no matching
            if len(anc_goal_obs) == 0:
                continue

            # Concatenate anchor
            anc_goal_obs = th.cat(anc_goal_obs, dim=0)
            anc_goal_next_obs = th.cat(anc_goal_next_obs, dim=0)

            # Concatenate positive
            pos_goal_obs = th.cat(pos_goal_obs, dim=0)
            pos_goal_next_obs = th.cat(pos_goal_next_obs, dim=0)

            # Concatenate negative
            neg_goal_obs = th.cat(neg_goal_obs, dim=0)
            neg_goal_next_obs = th.cat(neg_goal_next_obs, dim=0)

            # Get sampler
            ndata = len(anc_goal_obs)
            sampler = SubsetRandomSampler(range(ndata))
            sampler = BatchSampler(sampler, batch_size=max_batch_size, drop_last=False)

            # Get misc
            rand_inds = th.randint(len(obs_s), (ndata,))
            obs = obs_s[rand_inds]
            old_states = old_states_s[rand_inds]
            old_vtargs = old_vtargs_s[rand_inds]

            # Sample batch
            for inds in sampler:
                batch = {
                    # Anchor
                    "anc_goal_obs": anc_goal_obs[inds].cuda(),
                    "anc_goal_next_obs": anc_goal_next_obs[inds].cuda(),
                    # Positive
                    "pos_goal_obs": pos_goal_obs[inds].cuda(),
                    "pos_goal_next_obs": pos_goal_next_obs[inds].cuda(),
                    # Negative
                    "neg_goal_obs": neg_goal_obs[inds].cuda(),
                    "neg_goal_next_obs": neg_goal_next_obs[inds].cuda(),
                    # Misc
                    "obs": obs[inds].cuda(),
                    "old_states": old_states[inds].cuda(),
                    "old_vtargs": old_vtargs[inds].cuda(),
                }
                yield batch


class PPOADAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        model: PPOADModel,
        ppo_nepoch: int,
        ppo_nbatch: int,
        clip_param: float,
        vf_loss_coef: float,
        ent_coef: float,
        lr: float,
        max_grad_norm: float,
        aux_freq: int,
        aux_nepoch: int,
        pi_dist_coef: int,
        vf_dist_coef: int,
    ):
        super().__init__(model)
        self.model: PPOADModel

        # PPO params
        self.ppo_nepoch = ppo_nepoch
        self.ppo_nbatch = ppo_nbatch
        self.clip_param = clip_param
        self.vf_loss_coef = vf_loss_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_count = 0

        # Aux params
        self.aux_freq = aux_freq
        self.aux_nepoch = aux_nepoch
        self.pi_dist_coef = pi_dist_coef
        self.vf_dist_coef = vf_dist_coef

        # Buffer
        self.buffer = Buffer(maxlen=aux_freq)

        # Optimizers
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.match_optimizer = optim.Adam(model.parameters(), lr=lr)
        self.pred_optimizer = optim.Adam(model.parameters(), lr=lr)

    def update(self, storage: RolloutStorage):
        # Set model to training mode
        self.model.train()

        # Insert data to buffer
        keys = ["obs", "actions", "states", "returns", "masks", "successes"]
        seg = {key: storage[key].cpu() for key in keys}
        self.buffer.insert(seg)

        # Run PPO
        pi_loss_epoch = 0
        vf_loss_epoch = 0
        entropy_epoch = 0
        nupdate = 0

        for _ in range(self.ppo_nepoch):
            # Get data loader
            data_loader = storage.get_data_loader(self.ppo_nbatch)

            for batch in data_loader:
                # Compute loss
                losses = self.model.compute_losses(**batch, clip_param=self.clip_param)
                pi_loss = losses["pi_loss"]
                vf_loss = losses["vf_loss"]
                entropy = losses["entropy"]
                loss = pi_loss + self.vf_loss_coef * vf_loss - self.ent_coef * entropy

                # Update parameter
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update stats
                pi_loss_epoch += pi_loss.item()
                vf_loss_epoch += vf_loss.item()
                entropy_epoch += entropy.item()
                nupdate += 1

        # Compute average stats
        pi_loss_epoch /= nupdate
        vf_loss_epoch /= nupdate
        entropy_epoch /= nupdate

        # Define train stats
        train_stats = {
            "pi_loss": pi_loss_epoch,
            "vf_loss": vf_loss_epoch,
            "entropy": entropy_epoch,
        }

        # Increase PPO count
        self.ppo_count += 1

        if self.ppo_count % self.aux_freq == 0:
            # Pre-process buffer
            self.buffer.parse_segs()
            self.buffer.preprocess_trajs()

            # Copy model and set it to eval mode
            old_model = copy.deepcopy(self.model)
            old_model.eval()

            # Run aux phase
            match_loss_epoch = 0
            pred_loss_epoch = 0
            pi_dist_epoch = 0
            vf_dist_epoch = 0
            match_nupdate = 0
            pred_nupdate = 0

            for _ in range(self.aux_nepoch):
                # Get match data loader
                match_data_loader = self.buffer.get_match_data_loader(self.model)

                for batch in match_data_loader:
                    # Compute match loss
                    match_losses = self.model.compute_match_losses(
                        **batch,
                        old_model=old_model,
                    )
                    match_loss = match_losses["match_loss"]
                    pi_dist = match_losses["pi_dist"]
                    vf_dist = match_losses["vf_dist"]
                    loss = (
                        match_loss
                        + self.pi_dist_coef * pi_dist
                        + self.vf_dist_coef * vf_dist
                    )

                    # Update parameters
                    self.match_optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.match_optimizer.step()

                    # Update stats
                    match_loss_epoch += match_loss.item()
                    pi_dist_epoch += pi_dist.item()
                    vf_dist_epoch += vf_dist.item()
                    match_nupdate += 1

                # Get pred data loader
                pred_data_loader = self.buffer.get_pred_data_loader()

                for batch in pred_data_loader:
                    # Compute pred loss
                    pred_losses = self.model.compute_pred_losses(
                        **batch,
                        old_model=old_model,
                    )
                    pred_loss = pred_losses["pred_loss"]
                    pi_dist = pred_losses["pi_dist"]
                    vf_dist = pred_losses["vf_dist"]
                    loss = (
                        pred_loss
                        + self.pi_dist_coef * pi_dist
                        + self.vf_dist_coef * vf_dist
                    )

                    # Update parameters
                    self.pred_optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.pred_optimizer.step()

                    # Update stats
                    pred_loss_epoch += pred_loss.item()
                    pi_dist_epoch += pi_dist.item()
                    vf_dist_epoch += vf_dist.item()
                    pred_nupdate += 1

            # Compute average stats
            match_loss_epoch /= match_nupdate
            pred_loss_epoch /= pred_nupdate
            pi_dist_epoch /= match_nupdate + pred_nupdate
            vf_dist_epoch /= match_nupdate + pred_nupdate

            # Define aux train stats
            aux_train_stats = {
                "match_loss": match_loss_epoch,
                "pred_loss": pred_loss_epoch,
                "pi_dist": pi_dist_epoch,
                "vf_dist": vf_dist_epoch,
            }

            # Update train stats
            train_stats.update(aux_train_stats)

        return train_stats
