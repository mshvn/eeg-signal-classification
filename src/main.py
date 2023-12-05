# /Users/mike/miniconda3/envs/eeg

# conda info --envs
# poetry shell
# poetry env list

# black src/*.py
# isort src/*.py
# flake8 src
# pre-commit installed w/conda
# pre-commit run --all-files

# (eeg-catboost-py3.12) ➜  EEG_catboost deactivate
# (base) ➜  EEG_catboost

import fire
from hydra import compose, initialize
from omegaconf import OmegaConf

from infer import infer_model
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


# defaults are 20 0.5 7

# train_model(df_all_str, iterations, learning_rate, depth)

# train_model(
# df_all_str="../data/raw/sp1s_aa_train.txt",
# iterations=10,
# learning_rate=0.5,
# depth=7
# )


# if __name__ == "__main__":
#     cbc_eeg = Cbc()
#     fire.Fire(cbc_eeg)


# if __name__ == "__main__":
#     main()


# !pip install catboost

# import pandas as pd
# import numpy as np
# from tqdm import tqdm

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report

# # from sklearn.model_selection import GridSearchCV
# from catboost import CatBoostClassifier


# df_all = pd.read_fwf('../data/raw/sp1s_aa_train.txt', header=None)
# df_eval = pd.read_fwf('../data/raw/sp1s_aa_test.txt', header=None)

# df_all.head()

# print(df_all.shape, df_eval.shape)

# df_all_X = df_all.drop(0, axis=1)
# df_all_y = df_all[0].astype(int)

# train test split

# X_train, X_test, y_train, y_test =
# train_test_split(df_all_X, df_all_y,
# test_size=0.05, shuffle=False, random_state=42)

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

"""---"""

# model = CatBoostClassifier(iterations=20,
#                            learning_rate=0.5,
#                            depth=7)


# model.fit(X_train, y_train)

# preds_class = model.predict(X_test)
# Get predicted probabilities for each class
# preds_proba = model.predict_proba(X_test)
# Get predicted RawFormulaVal
# preds_raw = model.predict(X_test, prediction_type='RawFormulaVal')

# from sklearn.metrics import accuracy_score

# print(accuracy_score(y_test, preds_class))


# print(list(preds_class[:15]), list(y_test[:15]))

# target_names = ['class 0', 'class 1']
# print(classification_report(y_test, preds_class, target_names=target_names))

# print('EVAL:')

# X_test.shape, df_eval.shape

# print(model.predict(df_eval))
