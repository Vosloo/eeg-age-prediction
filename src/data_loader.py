from pathlib import Path
from typing import Generator

import pandas as pd
from src.model.measurement import Measurement

DATA_DIR = Path(__file__).parent.parent / "data" / "parquet"
EVAL_DIR = DATA_DIR / "eval"
TRAIN_DIR = DATA_DIR / "train"
EVAL_AGE = DATA_DIR / "eval.csv"
TRAIN_AGE = DATA_DIR / "train.csv"


class DataLoader:
    @staticmethod
    def get_eval(limit: int) -> list[Measurement]:
        batch = []
        ages = pd.read_csv(EVAL_AGE, index_col=0)
        for file in EVAL_DIR.iterdir():
            age = ages.loc[file.name, "age"]
            batch.append(Measurement(age, file))

            if len(batch) == limit:
                return batch

    @staticmethod
    def get_train_iter(batch_size: int) -> Generator[list[Measurement], None, None]:
        batch = []
        ages = pd.read_csv(TRAIN_AGE, index_col=0)
        for file in TRAIN_DIR.iterdir():
            age = ages.loc[file.name, "age"]
            batch.append(Measurement(age, file))

            if len(batch) == batch_size:
                yield batch
                batch = []
