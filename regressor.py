from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

MODEL_PATH = Path("xgbRegressor.bin")
SCALER_PATH = Path("out/StandardScaler.joblib")


def standardize_data(data: np.array) -> np.array:
    print("Standardizing data...")
    if not SCALER_PATH.exists():
        scaler = StandardScaler()
        scaler.fit(data)
        # dump(scaler, SCALER_PATH)
    else:
        scaler = load(SCALER_PATH)

    return scaler.transform(data)


def train_regressor():
    print("Loading data...")
    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        eta=0.02,
        n_jobs=6,
        verbosity=1,
        early_stopping_rounds=100,
    )
    train_dataset = pd.read_csv("out/fft_train.csv")
    X = standardize_data(train_dataset.iloc[:, :-1].values)
    # X = train_dataset.iloc[:, :-1].values
    y = train_dataset.iloc[:, -1]

    eval_dataset = pd.read_csv("out/fft_eval.csv")
    eval_X = standardize_data(eval_dataset.iloc[:, :-1].values)
    # eval_X = eval_dataset.iloc[:, :-1].values
    eval_y = eval_dataset.iloc[:, -1]

    model.fit(X, y, eval_set=[(eval_X, eval_y)])

    for i, (true, pred) in enumerate(zip(eval_y, model.predict(eval_X))):
        print(f"True: {true:.2f}, Predicted: {pred:.2f}")


def train_svr():
    print("Loading data...")
    model = SVR(
        kernel="poly",
        verbose=True,
    )
    train_dataset = pd.read_csv("out/fft_train.csv")
    X = standardize_data(train_dataset.iloc[:, :-1].values)
    # X = train_dataset.iloc[:, :-1].values
    y = train_dataset.iloc[:, -1]

    eval_dataset = pd.read_csv("out/fft_eval.csv")
    eval_X = standardize_data(eval_dataset.iloc[:, :-1].values)
    # eval_X = eval_dataset.iloc[:, :-1].values
    eval_y = eval_dataset.iloc[:, -1]

    model.fit(X, y)

    for i, (true, pred) in enumerate(zip(eval_y, model.predict(eval_X))):
        print(f"True: {true:.2f}, Predicted: {pred:.2f}")


if __name__ == "__main__":
    train_regressor()
    # train_svr()
