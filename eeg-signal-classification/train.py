import os

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def start_train(df_all_str, iterations, learning_rate, depth):
    print("Pulling with DVC:")
    os.system("dvc pull")

    print("-= TRAIN starts =-")
    df_all = pd.read_fwf(df_all_str, header=None)

    df_all_X = df_all.drop(0, axis=1)
    df_all_y = df_all[0].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df_all_X, df_all_y, test_size=0.05, shuffle=False, random_state=42
    )

    model = CatBoostClassifier(
        iterations=iterations, learning_rate=learning_rate, depth=depth
    )

    model.fit(X_train, y_train)

    preds_class = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds_class)
    print("Accuracy:", accuracy)

    logloss = model.get_evals_result()
    print("Log loss:", logloss["learn"]["Logloss"][-1])

    print("Saving model...")
    model.save_model("../models/catboost/cbc.cbm", format="cbm")
    print("Done.")

    return (
        accuracy,
        logloss["learn"]["Logloss"][-1],
        model.get_best_score()["learn"]["Logloss"],
    )
