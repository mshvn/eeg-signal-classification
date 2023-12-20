import pandas as pd
from catboost import CatBoostClassifier


def infer_model(df_eval_str, model_str):
    print("INFERENCE starts")
    df_eval = pd.read_fwf(df_eval_str, header=None)
    model_str = model_str

    from_file = CatBoostClassifier()
    from_file.load_model(model_str)

    print("Prediction:")
    print(from_file.predict(df_eval))

    return


# print('EVAL:')
# print(df_eval.shape)
# print(model.predict(df_eval))
# df_eval = pd.read_fwf('../data/raw/sp1s_aa_test.txt', header=None)
