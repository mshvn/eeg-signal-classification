# python3 commands.py train
# python3 commands.py infer

import fire
from hydra import compose, initialize
from infer import infer_model
from omegaconf import OmegaConf
from train import train_model

initialize(version_base=None, config_path="../config", job_name="cbc_app")
cfg = compose(config_name="config", overrides=[])


def train():
    print("Hydra params (config.yaml):\n", OmegaConf.to_yaml(cfg))

    train_model(
        cfg.params.df_all_str,
        cfg.params.iterations,
        cfg.params.learning_rate,
        cfg.params.depth,
    )
    return


def infer():
    infer_model(cfg.params.df_eval_str, cfg.params.model_str)
    return


if __name__ == "__main__":
    fire.Fire({"train": train, "infer": infer})
