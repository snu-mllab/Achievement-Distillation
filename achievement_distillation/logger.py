import logging
from pathlib import Path
from typing import Dict, Optional


class Logger:
    def __init__(
        self,
        config: Dict,
        group: str,
        name: str,
        use_wandb: bool = True,
        use_python_logger: bool = False,
        out_dir: Optional[str] = None,
    ):
        if out_dir is not None:
            path = Path(out_dir)
            path.mkdir(parents=True, exist_ok=True)
        if use_wandb:
            import wandb

            self.wandb_writer = wandb.init(
                config=config,
                project="crafter",
                group=group,
                name=name,
            )
        if use_python_logger:
            self.logger = logging.getLogger(name)
            if out_dir is None:
                handler = logging.StreamHandler()
            else:
                name_ = name.replace("/", "_")
                handler = logging.FileHandler(f"{out_dir}/{name_}.log")

            formatter = logging.Formatter(
                "%(asctime)s %(levelname)-8s %(message)s", datefmt="%x %X"
            )
            handler.setFormatter(formatter)

            self.logger.setLevel(level=logging.INFO)
            self.logger.addHandler(handler)

        self.wandb = use_wandb
        self.python_logger = use_python_logger

    def log(self, msg_dict: Dict, step: int):
        if self.wandb:
            self.wandb_writer.log(msg_dict, step=step)

        if self.python_logger:
            for k, v in msg_dict.items():
                self.logger.info(f"Step {step}: {k}: {v}")
