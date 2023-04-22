from pathlib import Path

import pandas as pd
import regex as re

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
    def __init__(self, age_line: str, file: Path) -> None:
        self.age = self._extract_age(age_line)
        self.file = file
        self._data = None

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self._data = pd.read_csv(self.file, skiprows=1)
            self._data.columns = [item.strip("# ") for item in self._data.columns]
            self._data = self._data[KEEP_COLUMNS]

        return self._data

    def _extract_age(self, age_line: str) -> int:
        return int(re.search(r"\d+", age_line).group(0))

    def __repr__(self) -> str:
        return f"Measurement(age={self.age}, file={'/'.join(self.file.parts[-2:])})"

    def __str__(self) -> str:
        return self.__repr__()
