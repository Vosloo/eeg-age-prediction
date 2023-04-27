from pathlib import Path
from typing import Generator

import pandas as pd
from src.model.measurement import Measurement

DATA_DIR = Path(__file__).parent.parent / "data" / "parquet"
EVAL_DIR = DATA_DIR / "eval"
TRAIN_DIR = DATA_DIR / "train"
EVAL_AGE = DATA_DIR / "eval_age.csv"
TRAIN_AGE = DATA_DIR / "train_age.csv"

TRAIN_STATS_PATH = Path("out/train_stats.csv")
EVAL_STATS_PATH = Path("out/eval_stats.csv")

class DataLoader:
    eval_size = len(list(EVAL_DIR.iterdir()))
    train_size = len(list(TRAIN_DIR.iterdir()))

    @staticmethod
    def get_eval(limit: int = -1) -> list[Measurement]:
        batch = []
        ages = pd.read_csv(EVAL_AGE, index_col=0)
        for file in EVAL_DIR.iterdir():
            age = ages.loc[file.name, "age"]
            batch.append(Measurement(age, file))

            if limit != -1 and len(batch) == limit:
                return batch
        
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

        if batch:
            yield batch

    @staticmethod
    def get_statistic_train() -> pd.DataFrame:
        if not TRAIN_STATS_PATH.exists():
            raise FileNotFoundError("Train statistics not found. Run statistifier.ipynb to generate them.")

        return pd.read_csv(TRAIN_STATS_PATH)
    

    @staticmethod
    def get_statistic_eval() -> pd.DataFrame:
        if not EVAL_STATS_PATH.exists():
            raise FileNotFoundError("Eval statistics not found. Run statistifier.ipynb to generate them.")

        return pd.read_csv(EVAL_STATS_PATH)
