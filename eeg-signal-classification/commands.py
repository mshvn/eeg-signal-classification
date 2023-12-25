# Please follow README.md instructions to run the project

import fire
import git
import mlflow
from hydra import compose, initialize
from infer import start_infer

# from omegaconf import OmegaConf # ← uncomment to use OmegaConf
from train import start_train


def train():
    """Train entry point"""

    mlflow.set_tracking_uri(uri=cfg.mlops.mlflow_server)
    mlflow.set_experiment("mlflow_train")

    repo = git.Repo(search_parent_directories=True)
    commit_id = repo.head.object.hexsha

    # ↓ uncomment to print config.yaml:
    # print("Hydra params (config.yaml):\n", OmegaConf.to_yaml(cfg))

    with mlflow.start_run():
        params = {
            "Iterations": cfg.params.iterations,
            "Learning rate": cfg.params.learning_rate,
            "Depth": cfg.params.depth,
            "Commit id": commit_id[:],
        }

        mlflow.log_params(params)

        accuracy, log_loss, best_score = start_train(
            cfg.params.df_all_str,
            cfg.params.iterations,
            cfg.params.learning_rate,
            cfg.params.depth,
        )

        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Log loss", log_loss)
        mlflow.log_metric("Best Score", best_score)
        mlflow.set_tag("Training Info", "Catboost model")
    return


def infer():
    """Infer entry point"""

    start_infer(cfg.params.df_eval_str, cfg.params.model_str)
    return


if __name__ == "__main__":
    initialize(version_base=None, config_path="../config", job_name="cbc_app")
    cfg = compose(config_name="config", overrides=[])
    fire.Fire({"train": train, "infer": infer})
