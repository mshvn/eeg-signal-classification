import pandas as pd
from catboost import CatBoostClassifier


def start_infer(df_eval_str, model_str):
    print("-= INFERENCE starts =-")
    df_eval = pd.read_fwf(df_eval_str, header=None)

    cb_cls = CatBoostClassifier()
    cb_cls.load_model(model_str)

    # print(from_file.predict(df_eval))

    out = cb_cls.predict(df_eval)
    print("Writing to output...")
    out.tofile("../result/out.csv", sep=",")
    print("Done.")

    return
