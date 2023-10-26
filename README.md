# Achievement Distillation

This is the code for the paper [Discovering Hierarchical Achievements in Reinforcement Learning via Contrastive Learning](https://arxiv.org/abs/2307.03486) accepted to NeurIPS 2023.

<img src="figures/overview.png" width="800">
<img src="figures/result.png" width="600">

## Installation

```
conda create --name ad-crafter python=3.10
conda activate ad-crafter

pip install --upgrade "setuptools==65.7.0" "wheel==0.38.4"
pip install -r requirements.txt
pip install -e .
```

## Usage

To execute `train.py` in a Bash environment, you can use the following commands. By default, the system will assign the timestamp `debug`. 

PPO
```
python train.py --exp_name ppo --log_stats
```

PPO + Achievement Distillation (ours)
```
python train.py --exp_name ppo_ad --log_stats
```

If you are working in an environment that uses the Slurm Workload Manager, you can submit your job using `slurm.py`. In this setup, the system automatically assigns a timestamp that corresponds to the actual start time of your job.


## Citation

If you find this code useful, please cite this work.

```
@inproceedings{moon2023ad,
    title={Discovering Hierarchical Achievements in Reinforcement Learning via Contrastive Learning},
    author={Seungyong Moon and Junyoung Yeom and Bumsoo Park and Hyun Oh Song},
    booktitle={Neural Information Processing Systems},
    year={2023}
}
```

## Credit
- https://github.com/openai/Video-Pre-Training
- https://github.com/snu-mllab/DCPG
