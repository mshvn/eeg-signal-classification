# import numpy as np
import pandas as pd
# from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
# , classification_report
from sklearn.model_selection import train_test_split

# from tqdm import tqdm


def train_model(df_all_str, iterations, learning_rate, depth):
    print("TRAIN starts")
    # df_all_str is string
    df_all = pd.read_fwf(df_all_str, header=None)

    print("DataFrame shape:", df_all.shape)

    df_all_X = df_all.drop(0, axis=1)
    df_all_y = df_all[0].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df_all_X, df_all_y, test_size=0.05, shuffle=False, random_state=42
    )
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    model = CatBoostClassifier(
        iterations=iterations, learning_rate=learning_rate, depth=depth
    )  # 20 0.5 7

    model.fit(X_train, y_train)

    preds_class = model.predict(X_test)
    preds_proba = model.predict_proba(X_test)

    print("Pred. classes:\n", preds_class[:9])
    print("Pred. proba:\n", preds_proba[:9])

    print("Accuracy:", accuracy_score(y_test, preds_class))
    print("Actual:   ", list(y_test[:9]))
    print("Predicted:", list(preds_class[:9]))

    # target_names = ['class 0', 'class 1']
    # print(classification_report(y_test, preds_class,
    # target_names=target_names))

    print("Saving model...")
    model.save_model("../models/catboost/cbc.cbm", format="cbm")

    return
