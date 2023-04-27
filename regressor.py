from pathlib import Path

import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor

from src.data_loader import DataLoader
from src.model.measurement import Measurement

MODEL_PATH = Path("xgbRegressor.bin")


if __name__ == "__main__":
    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        eta=0.1,
        n_jobs=6,
        verbosity=1,
        early_stopping_rounds=20,
    )

    train_dataset = DataLoader.get_statistic_train()
    X = train_dataset.drop(columns=["age"])
    y = train_dataset["age"]

    eval_dataset = DataLoader.get_statistic_eval()
    eval_X = eval_dataset.drop(columns=["age"])
    eval_y = eval_dataset["age"]

    model.fit(X, y, eval_set=[(eval_X, eval_y)])

    for i, (true, pred) in enumerate(zip(eval_y, model.predict(eval_X))):
        print(f"True: {true:.2f}, Predicted: {pred:.2f}")
