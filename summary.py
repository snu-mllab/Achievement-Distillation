import argparse
import json
import os

import numpy as np
import pandas as pd

from achievement_distillation.constant import TASKS


def main(args):
    timesteps = np.arange(10000, 10010000, 10000)
    total_success_rate = np.zeros((args.num_seeds, 1000, len(TASKS)))
    total_scores = np.zeros((args.num_seeds, 1000))
    total_rewards = np.zeros((args.num_seeds, 1000))

    for seed in range(args.num_seeds):
        counts = []
        rewards = []
        scores = []
        success_rates = []
        successes = np.zeros((0, len(TASKS)), dtype=np.int32)

        run_name = f"{args.exp_name}-s{seed:02}"
        log_path = os.path.join(args.log_dir, run_name, "stats.jsonl")

        log_file = open(log_path, "r")
        trajs = [json.loads(line) for line in log_file]

        count = 0

        for traj in trajs:
            length = traj["length"]
            count += length
            counts.append(count)

            reward = traj["reward"]
            rewards.append(reward)
            
            achievement = []
            for task in TASKS:
                achievement.append(traj[f"achievement_{task}"])
            achievement = np.array(achievement, dtype=np.int32)[np.newaxis, ...]
            
            success = (achievement > 0).astype(np.int32)
            successes = np.concatenate([successes, success], axis=0)
            
            success_rate = 100 * np.mean(successes, axis=0)
            success_rates.append(success_rate)

            score = np.exp(np.mean(np.log(1 + success_rate))) - 1
            scores.append(score)

        
        timesteps = np.array(timesteps, dtype=np.int32)
        scores = np.array(scores, dtype=np.float32)
        success_rates = np.stack(success_rates, axis=0)

        for i, timestep in enumerate(timesteps):
            inds, = np.where(counts <= timestep)
            ind = inds[-1]
            inds_prev, = np.where(counts <= timestep - 100000)
            if timestep <= 100000:
                ind_prev = 0
            else:
                ind_prev = inds_prev[-1]
            total_scores[seed, i] = scores[ind]
            total_rewards[seed, i] = np.mean(rewards[max(ind_prev, 0):ind + 1])
            total_success_rate[seed, i] = success_rates[ind]
    
    scores_mean = np.mean(total_scores, axis=0)
    scores_std = np.std(total_scores, axis=0)
    rewards_mean = np.mean(total_rewards, axis=0)
    rewards_std = np.std(total_rewards, axis=0)

    score_data = {
        "timesteps": timesteps,
        "scores_mean": scores_mean,
        "scores_std": scores_std,
        "scores_mean_plus_std": scores_mean + scores_std,
        "scores_mean_minus_std": scores_mean - scores_std,
        "rewards_mean": rewards_mean,
        "rewards_std": rewards_std,
        "rewards_mean_plus_std": rewards_mean + rewards_std,
        "rewards_mean_minus_std": rewards_mean - rewards_std,
    }

    score_csv_path = os.path.join(args.data_dir, f"{args.exp_name}-score.csv")
    score_df = pd.DataFrame(score_data)
    score_df.to_csv(score_csv_path, index=False)
    
    total_success_rate_mean = np.mean(total_success_rate, axis=0)

    success_rate_data = {
        "task": [task.replace("_", " ").capitalize() for task in TASKS],
        "success_rate": total_success_rate_mean[-1],
    }
    
    success_rate_csv_path = os.path.join(args.data_dir, f"{args.exp_name}-success_rate.csv")
    success_rate_df = pd.DataFrame(success_rate_data)
    success_rate_df.to_csv(success_rate_csv_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    main(args)
