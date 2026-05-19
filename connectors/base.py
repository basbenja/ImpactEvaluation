from abc import ABC, abstractmethod

import numpy as np

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


Record = tuple[int, np.ndarray, int, int]  # (firm_id, sequence, cohort_id, label)


class BaseConnector(ABC):
    @abstractmethod
    def build_train_records(self) -> list[Record]: ...

    @abstractmethod
    def build_test_records(self) -> list[Record]: ...

    @abstractmethod
    def fit_scaler(self, records: list[Record]) -> StandardScaler: ...

    @abstractmethod
    def scale_records(
        self, records: list[Record], scaler: StandardScaler
    ) -> list[Record]: ...

    @abstractmethod
    def convert(self, fit_scaler: bool = True) -> tuple[Dataset, Dataset]: ...
