import argparse
import datetime
from functools import partial
import json
import os
import random
import sys
import yaml

import numpy as np
import torch as th

from crafter.env import Env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from achievement_distillation.algorithm import *
from achievement_distillation.constant import TASKS
from achievement_distillation.logger import Logger
from achievement_distillation.model import *
from achievement_distillation.sample import sample_rollouts
from achievement_distillation.storage import RolloutStorage
from achievement_distillation.wrapper import VecPyTorch


def main(args):
    # Load config file
    config_file = open(f"configs/{args.exp_name}.yaml", "r")
    config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)
    th.backends.cudnn.benchmark = False

    # CUDA setting
    th.set_num_threads(1)
    cuda = th.cuda.is_available()
    device = th.device("cuda:0" if cuda else "cpu")

    # Create logger
    group_name = f"{args.exp_name}-{args.timestamp}"
    run_name = f"{group_name}-s{args.seed:02}"

    if args.log_stats:
        # JSON
        log_dir = os.path.join("./logs", run_name)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "stats.jsonl")
        log_file = open(log_path, "w")

        # W&B
        logger = Logger(config=config, group=group_name, name=run_name)

    # Create checkpoint directory
    if args.save_ckpt:
        ckpt_dir = os.path.join("./models", run_name)
        os.makedirs(ckpt_dir, exist_ok=True)

    # Create environment
    seeds = np.random.randint(0, 2**31 - 1, size=config["nproc"])
    env_fns = [partial(Env, seed=seed) for seed in seeds]
    venv = SubprocVecEnv(env_fns)
    venv = VecMonitor(venv)
    venv = VecPyTorch(venv, device=device)
    obs = venv.reset()

    # Create storage
    storage = RolloutStorage(
        nstep=config["nstep"],
        nproc=config["nproc"],
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        hidsize=config["model_kwargs"]["hidsize"],
        device=device,
    )
    storage.obs[0].copy_(obs)

    # Create model
    model_cls = getattr(sys.modules[__name__], config["model_cls"])
    model: BaseModel = model_cls(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        **config["model_kwargs"],
    )
    model = model.to(device)
    print(model)

    # Create algorithm
    algorithm_cls = getattr(sys.modules[__name__], config["algorithm_cls"])
    algorithm: BaseAlgorithm = algorithm_cls(
        model=model,
        **config["algorithm_kwargs"],
    )

    # Run algorithm
    total_successes = np.zeros((0, len(TASKS)), dtype=np.int32)

    for epoch in range(1, config["nepoch"] + 1):
        # Sample episodes
        rollout_stats = sample_rollouts(venv, model, storage)

        # Compute returns
        storage.compute_returns(config["gamma"], config["gae_lambda"])

        # Update models
        train_stats = algorithm.update(storage)

        # Reset storage
        storage.reset()

        # Compute score
        successes = rollout_stats["successes"]
        total_successes = np.concatenate([total_successes, successes], axis=0)
        success_rate = 100 * np.mean(total_successes, axis=0)
        score = np.exp(np.mean(np.log(1 + success_rate))) - 1

        # Get eval stats
        eval_stats = {
            "success_rate": {k: v for k, v in zip(TASKS, success_rate)},
            "score": score,
        }

        # Print stats
        print(f"\nepoch {epoch}:")
        print(json.dumps(train_stats, indent=2))
        print(json.dumps(eval_stats, indent=2))

        # Log stats
        if args.log_stats:
            # JSON
            episode_lengths = rollout_stats["episode_lengths"]
            episode_rewards = rollout_stats["episode_rewards"]
            achievements = rollout_stats["achievements"]

            for i in range(len(episode_lengths)):
                rollout_stat = {
                    "length": int(episode_lengths[i]),
                    "reward": round(float(episode_rewards[i]), 1),
                }
                for j, task in enumerate(TASKS):
                    rollout_stat[f"achievement_{task}"] = int(achievements[i, j])

                log_file.write(json.dumps(rollout_stat) + "\n")
                log_file.flush()

            # W&B
            logger.log(train_stats, epoch)
            logger.log(eval_stats, epoch)

        # Save checkpoint
        if args.save_ckpt and epoch % config["save_freq"] == 0:
            ckpt_path = os.path.join(ckpt_dir, f"agent-e{epoch:03}.pt")
            th.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--timestamp", type=str, default="debug")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_stats", action="store_true")
    parser.add_argument("--save_ckpt", action="store_true")
    args = parser.parse_args()

    # Run main
    main(args)
