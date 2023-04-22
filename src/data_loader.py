from pathlib import Path
from typing import Generator

import pandas as pd

from src.model.measurement import Measurement

DATA_DIR = Path(__file__).parent.parent / "data"
EVAL_DIR = DATA_DIR / "eval"
TRAIN_DIR = DATA_DIR / "train"


class DataLoader:
    @staticmethod
    def get_eval(limit: int) -> list[Measurement]:
        batch = []
        no_files = len(list(EVAL_DIR.iterdir()))
        print("Loading eval data...")
        for ind, file in enumerate(EVAL_DIR.iterdir()):
            print(f"\r{ind + 1:3}/{no_files:3}", end="")
            with open(file, "r") as f:
                age_line = f.readline()

            batch.append(Measurement(age_line, file))
            if len(batch) == limit:
                print("\nDone")
                return batch

    @staticmethod
    def get_train_iter(batch_size: int) -> Generator[list[Measurement], None, None]:
        batch = []
        no_files = len(list(TRAIN_DIR.iterdir()))
        for ind, file in enumerate(TRAIN_DIR.iterdir()):
            print(f"\r{ind + 1:4}/{no_files:4}", end="")
            with open(file, "r") as f:
                age_line = f.readline()
            
            batch.append(Measurement(age_line, file))
            if len(batch) == batch_size:
                yield batch
                batch = []
