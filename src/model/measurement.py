from pathlib import Path

import pandas as pd

KEEP_COLUMNS = [
    "EEG FP1-REF",
    "EEG FP2-REF",
    "EEG F3-REF",
    "EEG F4-REF",
    "EEG C3-REF",
    "EEG C4-REF",
    "EEG P3-REF",
    "EEG P4-REF",
    "EEG O1-REF",
    "EEG O2-REF",
    "EEG F7-REF",
    "EEG F8-REF",
    "EEG T3-REF",
    "EEG T4-REF",
    "EEG T5-REF",
    "EEG T6-REF",
    "EEG A1-REF",
    "EEG A2-REF",
    "EEG FZ-REF",
    "EEG CZ-REF",
    "EEG PZ-REF",
    "EEG EKG1-REF",
    "EEG T1-REF",
    "EEG T2-REF",
    "IBI",
    "BURSTS",
    "SUPPR",
]


class Measurement:
    def __init__(self, age: int, file: Path) -> None:
        self.age = age
        self.file = file
        self._data = None

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self._data = pd.read_parquet(self.file, columns=KEEP_COLUMNS)

        return self._data

    def __repr__(self) -> str:
        return f"Measurement(age={self.age}, file={'/'.join(self.file.parts[-2:])})"

    def __str__(self) -> str:
        return self.__repr__()

    def __lt__(self, other: "Measurement") -> bool:
        return self.age < other.age
    
    def __gt__(self, other: "Measurement") -> bool:
        return self.age > other.age
    
    def __eq__(self, other: "Measurement") -> bool:
        return self.age == other.age
