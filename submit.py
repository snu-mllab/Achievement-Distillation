import os
import datetime
import subprocess


def main():
    # WANDB
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    # Base command
    exp_name = "ppo_ad"
    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    base_cmd = f"python train.py --exp_name {exp_name} --timestamp {timestamp} --log_stats"

    # Submit jobs
    for seed in range(0, 10):
        cmd = f"{base_cmd} --seed {seed}"
        # Run sbatch command using subprocess


if __name__ == "__main__":
    main()
