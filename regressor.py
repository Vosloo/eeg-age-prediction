from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

from src.data_loader import DataLoader
from src.model.measurement import Measurement

MODEL_PATH = Path("xgbRegressor.bin")


if __name__ == "__main__":
    # model = XGBRegressor(
    #     n_estimators=200,
    #     max_depth=4,
    #     eta=0.1,
    #     n_jobs=6,
    #     verbosity=1,
    #     early_stopping_rounds=20,
    # )
    model = LinearRegression(n_jobs=6)

    # train_dataset = DataLoader.get_statistic_train()
    train_dataset = pd.read_csv("out/fft_train.csv")
    X = train_dataset.iloc[:, :-1]
    y = train_dataset.iloc[:, -1]

    eval_dataset = pd.read_csv("out/fft_eval.csv")
    eval_X = eval_dataset.iloc[:, :-1]
    eval_y = eval_dataset.iloc[:, -1]

    # model.fit(X, y, eval_set=[(eval_X, eval_y)])
    model.fit(X, y)

    for i, (true, pred) in enumerate(zip(eval_y, model.predict(eval_X))):
        print(f"True: {true:.2f}, Predicted: {pred:.2f}")
