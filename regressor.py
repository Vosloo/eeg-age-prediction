from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from src.data_loader import DataLoader
from src.model.measurement import Measurement

N_COMPONENTS = 600
FUN = "cube"

MODEL_PATH = Path("xgbRegressor.bin")
SCALER_PATH = Path("out/StandardScaler.joblib")
ICA_PATH = Path(f"out/ICA_{FUN}_{N_COMPONENTS}.joblib")


def standardize_data(data: np.array) -> np.array:
    print("Standardizing data...")
    if not SCALER_PATH.exists():
        scaler = StandardScaler()
        scaler.fit(data)
        dump(scaler, SCALER_PATH)
    else:
        scaler = load(SCALER_PATH)

    return scaler.transform(data)


def ica_data(data: np.array) -> np.array:
    print("ICA data...")
    if not ICA_PATH.exists():
        ica = FastICA(n_components=N_COMPONENTS, fun=FUN, tol=1e-3, max_iter=5000, whiten='unit-variance')
        ica.fit(data)
        dump(ica, ICA_PATH)
    else:
        ica = load(ICA_PATH)

    return ica.transform(data)

def train_regressor():
    print("Loading data...")
    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        eta=0.02,
        n_jobs=6,
        verbosity=1,
        early_stopping_rounds=100,
    )
    train_dataset = pd.read_csv("out/fft_train_full.csv")
    print("Pre ICA shape:", train_dataset.iloc[:, :-1].shape)
    X = ica_data(standardize_data(train_dataset.iloc[:, :-1].values))
    print("After ICA shape:", X.shape)
    y = train_dataset.iloc[:, -1]

    eval_dataset = pd.read_csv("out/fft_eval_full.csv")
    eval_X = ica_data(standardize_data(eval_dataset.iloc[:, :-1].values))
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
    print("Pre ICA shape:", train_dataset.iloc[:, :-1].shape)
    X = ica_data(standardize_data(train_dataset.iloc[:, :-1].values))
    print("After ICA shape:", X.shape)
    y = train_dataset.iloc[:, -1]

    eval_dataset = pd.read_csv("out/fft_eval.csv")
    eval_X = ica_data(standardize_data(eval_dataset.iloc[:, :-1].values))
    eval_y = eval_dataset.iloc[:, -1]

    model.fit(X, y)

    for i, (true, pred) in enumerate(zip(eval_y, model.predict(eval_X))):
        print(f"True: {true:.2f}, Predicted: {pred:.2f}")


if __name__ == "__main__":
    train_regressor()
    # train_svr()
