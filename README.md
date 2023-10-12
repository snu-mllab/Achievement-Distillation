# Achievement Distillation

This is the code for the paper [Discovering Hierarchical Achievements in Reinforcement Learning via Contrastive Learning](https://arxiv.org/abs/2307.03486) accepted to NeurIPS 2023.

<embed src="figures/overview.pdf" type="application/pdf" width="500">


## Installation

```
conda create --name ad-crafter python=3.10
conda activate ad-crafter

pip install --upgrade "setuptools==65.7.0" "wheel==0.38.4"
pip install -r requirements.txt
pip install -e .
```

## Usage

PPO (baseline)
```
python train.py --exp_name ppo --log_stats
```

PPO + Achievement Distillation (ours)
```
python train.py --exp_name ppo_ad --log_stats
```

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