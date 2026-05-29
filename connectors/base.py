from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


Record = tuple[int, np.ndarray, int, int]  # (firm_id, sequence, cohort_id, label)


@dataclass
class PreprocessedPanel:
    panel: pd.DataFrame
    feature_cols: list[str]   # expanded (numeric + indicator + dummies)
    n_numeric: int             # number of numeric cols (scaled); rest are indicators


class BaseConnector(ABC):

    @staticmethod
    def preprocess_panel(
        panel: pd.DataFrame,
        feature_cols: list[str],
    ) -> PreprocessedPanel:
        """
        Encode feature columns by dtype:
            - object / Categorical → one-hot (pd.get_dummies), no scaling
            - bool or int/int64 in {0,1} → cast to float32, no scaling
            - numeric → left as-is, will be scaled downstream

        Returns a PreprocessedPanel with the expanded panel and metadata
        needed by fit_scaler / scale_records.
        """
        if any(col not in panel.columns for col in feature_cols):
            missing = [c for c in feature_cols if c not in panel.columns]
            raise ValueError(
                f"feature_cols no encontradas en el panel: {missing}"
            )

        numeric_cols: list[str] = []
        indicator_cols: list[str] = []
        dummy_cols: list[str] = []

        df = panel.copy()

        for col in feature_cols:
            dtype = panel[col].dtype
            if pd.api.types.is_string_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
                dummies = pd.get_dummies(panel[col], prefix=col).astype(np.float32)
                df = pd.concat([df, dummies], axis=1)
                df.drop(columns=[col], inplace=True)
                dummy_cols.extend(dummies.columns.tolist())
            elif dtype == bool or panel[col].isin([0, 1]).all():
                df[col] = df[col].astype(np.float32)
                indicator_cols.append(col)
            else:
                numeric_cols.append(col)

        expanded_cols = numeric_cols + indicator_cols + dummy_cols
        return PreprocessedPanel(
            panel=df,
            feature_cols=expanded_cols,
            n_numeric=len(numeric_cols),
        )

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
