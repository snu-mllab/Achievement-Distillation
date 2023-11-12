import argparse
import os
import sys
import random
import yaml

import numpy as np
import torch as th
import torch.nn.functional as F

from crafter.env import Env
from crafter.recorder import VideoRecorder
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from achievement_distillation.algorithm import *
from achievement_distillation.model import *
from achievement_distillation.wrapper import VecPyTorch


def main(args):
    # Load config file
    config_file = open(f"configs/{args.exp_name}.yaml", "r")
    config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Fix Random seed
    random.seed(args.eval_seed)
    np.random.seed(args.eval_seed)
    th.manual_seed(args.eval_seed)
    th.cuda.manual_seed_all(args.eval_seed)

    # CUDA setting
    th.set_num_threads(1)
    cuda = th.cuda.is_available()
    device = th.device("cuda:0" if cuda else "cpu")

    # Define checkpoint directory
    run_name = f"{args.exp_name}-{args.timestamp}-s{args.train_seed:02}"
    ckpt_dir = os.path.join("./models", run_name)

    # Create environment
    env = Env(seed=args.eval_seed)
    env = VideoRecorder(env, directory=f"./videos/{run_name}")
    venv = DummyVecEnv([lambda: env])
    venv = VecPyTorch(venv, device=device)

    # Create model
    model_cls = getattr(sys.modules[__name__], config["model_cls"])
    model: BaseModel = model_cls(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        **config["model_kwargs"],
    )
    model.to(device)
    print(model)

    # Load checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"agent-e250.pt")
    state_dict = th.load(ckpt_path)
    model.load_state_dict(state_dict)

    # Eval
    model.eval()
    obs = venv.reset()
    states = th.zeros(1, config["model_kwargs"]["hidsize"]).to(device)

    while True:
        outputs = model.act(obs, states=states)
        latents = outputs["latents"]
        actions = outputs["actions"]
        obs, rewards, dones, _ = venv.step(actions)

        # Done
        if dones.any():
            break

        # Update states
        if (rewards > 0.1).any():
            with th.no_grad():
                next_latents = model.encode(obs)
            states = next_latents - latents
            states = F.normalize(states, dim=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="ppo_ad")
    parser.add_argument("--timestamp", type=str, default="debug")
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=123)
    args = parser.parse_args()

    main(args)
