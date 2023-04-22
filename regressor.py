from pathlib import Path

import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor

from src.data_loader import DataLoader
from src.model.measurement import Measurement

MODEL_PATH = Path("xgbRegressor.bin")


def get_min_no_rows(measurements: list[Measurement]) -> int:
    return min([measurement.data.shape[0] for measurement in measurements])


if __name__ == "__main__":
    model = XGBRegressor(
        n_estimators=1000,
        max_depth=8,
        eta=0.1,
        n_jobs=5,
        verbosity=1,
    )

    eval_list = DataLoader.get_eval(limit=5)
    min_no_rows = 240752 - 2  # get_min_no_rows(eval_list)
    X_eval = np.array([measurement.data.values[:min_no_rows].flatten().T for measurement in eval_list])
    y_eval = np.array([])
    for ind, measurement in enumerate(eval_list):
        if ind == min_no_rows:
            break

        y_eval = np.append(y_eval, measurement.age)

    print(X_eval.shape)
    print(y_eval.shape)

    evals = [(xgb.DMatrix(X_eval, label=y_eval), "eval")]

    train_iter = DataLoader.get_train_iter(batch_size=10)

    trained_model = model
    for measurements in train_iter:
        min_no_rows = 240752 - 2  # get_min_no_rows(measurements)
        X = np.array(
            [measurement.data.values[:min_no_rows].flatten().T for measurement in measurements]
        )
        y = np.array([])
        for ind, measurement in enumerate(measurements):
            if ind == min_no_rows:
                break

            y = np.append(y, measurement.age)

        print(X.shape)
        print(y.shape)

        trained_model = xgb.train(
            params=model.get_xgb_params(),
            dtrain=xgb.DMatrix(X, label=y),
            evals=evals,
            verbose_eval=1,
            xgb_model=MODEL_PATH if MODEL_PATH.exists() else None,
        )

        trained_model.save_model(MODEL_PATH)
