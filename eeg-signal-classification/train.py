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

    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # defaults are 20 0.5 7
    model = CatBoostClassifier(
        iterations=iterations, learning_rate=learning_rate, depth=depth
    )

    model.fit(X_train, y_train)

    preds_class = model.predict(X_test)
    # preds_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, preds_class)
    print("Accuracy:", accuracy)

    # print("Pred. classes:\n", preds_class[:9])
    # print("Pred. proba:\n", preds_proba[:9])
    # print("Actual:   ", list(y_test[:9]))
    # print("Predicted:", list(preds_class[:9]))

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
