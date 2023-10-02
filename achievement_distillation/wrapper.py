from typing import Dict, Sequence

import numpy as np
import torch as th

from gym import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper

from achievement_distillation.constant import TASKS


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv: VecEnv, device: str = "cuda"):
        super().__init__(venv)
        self.observation_space = self.transform_observation_space()
        self.device = device

    def transform_observation_space(self):
        obs_space = self.observation_space
        obs_shape = getattr(obs_space, "shape")
        obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
        new_obs_space = spaces.Box(low=0, high=1, shape=obs_shape)

        return new_obs_space

    def reset(self) -> th.Tensor:
        obs = self.venv.reset()
        obs = self.transform_obs(obs)

        return obs

    def step_async(self, actions: th.Tensor):
        actions = self.transform_actions(actions)
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        obs = self.transform_obs(obs)
        rewards = self.transform_rewards(rewards)
        dones = self.transform_dones(dones)
        infos = self.transform_infos(infos)

        return obs, rewards, dones, infos

    def transform_obs(self, obs: np.ndarray) -> th.Tensor:
        assert len(obs.shape) == 4
        obs = np.transpose(obs, (0, 3, 1, 2))
        obs = th.from_numpy(obs).float().to(self.device) / 255.0

        return obs

    def transform_rewards(self, rewards: np.ndarray) -> th.Tensor:
        assert len(rewards.shape) == 1
        rewards = rewards[:, np.newaxis]
        rewards = th.from_numpy(rewards).float().to(self.device)

        return rewards

    def transform_dones(self, dones: np.ndarray) -> th.Tensor:
        assert len(dones.shape) == 1
        dones = dones[:, np.newaxis]
        dones = th.from_numpy(dones).float().to(self.device)

        return dones

    def transform_infos(
        self, infos: Sequence[Dict[str, np.ndarray]]
    ) -> Dict[str, th.Tensor]:
        # Episode lengths and rewards
        episode_lengths = th.zeros(len(infos)).long().to(self.device)
        episode_rewards = th.zeros(len(infos)).float().to(self.device)

        for i, info in enumerate(infos):
            if "episode" in info:
                episode_lengths[i] = int(info["episode"]["l"])
                episode_rewards[i] = float(info["episode"]["r"])

        # Achievements
        achievements = [
            [info["achievements"][task] for task in TASKS] for info in infos
        ]
        achievements = np.array(achievements)
        achievements = th.from_numpy(achievements).long().to(self.device)

        # Successes
        successes = (achievements > 0).long()

        # Infos
        infos = {
            "episode_lengths": episode_lengths,
            "episode_rewards": episode_rewards,
            "achievements": achievements,
            "successes": successes,
        }

        return infos

    def transform_actions(self, actions: th.Tensor) -> np.ndarray:
        assert len(actions.shape) == 2
        actions = actions.squeeze(dim=-1)
        actions = actions.cpu().numpy()

        return actions
