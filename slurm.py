import os
import datetime


def main():
    # WANDB
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    # Base command
    exp_name = "ppo_ad"
    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    base_cmd = f"python train.py --log_stats --exp_name {exp_name} --timestamp {timestamp}"

    # Launch jobs
    for seed in range(0, 10):
        cmd = f"{base_cmd} --seed {seed}"
        # Write sbatch command here


if __name__ == "__main__":
    main()
