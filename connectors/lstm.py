import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from typing import Optional

from data_generation.panel_schema import Col


class PanelSequenceDataset(Dataset):
    """
    Dataset de PyTorch para secuencias de panel.

    Cada item es una tupla (sequence, cohort, label) donde:
        - sequence : tensor (T_k, F), donde T_k es el número de períodos
            pre-tratemiento y F es el número de features (esta forma es la que
            espera una LSTM por ejemplo)
        - cohort: tensor escalar (int), representa la cohorte asignada al
            item
        - label: tensor escalar (float). 1 si es tratado/control, 0 si es NiNi
    """

    def __init__(self, records: list[tuple[np.ndarray, int, int]]):
        """
        Args:
            records: lista de (secuencia, cohorte, label)
        """
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return PanelSequenceDataset(self.records[idx])

        seq, cohort, label = self.records[idx]
        return (
            torch.tensor(seq,    dtype=torch.float32),
            torch.tensor(cohort, dtype=torch.long),
            torch.tensor(label,  dtype=torch.float32),
        )

    @staticmethod
    def collate_fn(batch):
        """
        batch: lista de (sequence, cohort, label)
        """
        sequences, cohorts, labels = zip(*batch)

        # Tenemos que devolver los largo originales para que el modelo sepa hasta
        # dónde leer (esto después se le pasa a pack_padded_sequence)
        lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long)

        # pad_sequence apila y rellena con 0s hasta la longitud máxima del batch
        # sequences_padded: (batch_size, T_max, F)
        sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)

        return (
            sequences_padded,
            lengths,
            torch.stack(cohorts),
            torch.stack(labels),
        )


class LSTMConnector:
    """
    Conector que transforma el panel + split en Datasets de PyTorch.

    Args:
        panel        : DataFrame en formato long
        split        : dict generado por SplitGenerator
        feature_cols : columnas a usar como features
    """
    def __init__(
        self,
        panel: pd.DataFrame,
        split: dict,
        feature_cols: list[str]
    ):
        self.panel = panel
        self.split = split
        self.scaler: Optional[StandardScaler] = None

        if any(col not in panel.columns for col in feature_cols):
            raise ValueError(
                "Alguna(s) columna(s) de feature no se encuentran en el panel. "
                f"Columnas del panel: {panel.columns.tolist()}, columnas de "
                f"feature: {feature_cols}"
            )

        self.feature_cols = feature_cols

        # Índice para acceso rápido por firma
        self._firms = {
            fid: df for fid, df in self.panel.groupby(Col.ID_FIRMA)
        }

        self._cohorts_periods = sorted(
            self.panel.loc[self.panel[Col.TRATADO_EN_T], Col.T].unique().tolist()
        )

    def _build_sequence(self, firm_id: int, t_k: int) -> np.ndarray:
        """
        Extrae los períodos estrictamente anteriores a t_k para una firma.
        Si la firma tiene menos períodos de los esperados, hace padding
        hacia atrás con el primer valor conocido.

        Returns:
            array (t_k, F)
        """
        firm_data = self._firms[firm_id]
        # Recordemos que en t_k el tratamiento ya tuvo efecto
        pre_tr_periods = firm_data[firm_data['t'] < t_k][self.feature_cols].values.astype(np.float32)

        # Si no hay datos previos, devolvemos una secuencia de ceros
        if len(pre_tr_periods) == 0:
            # Esto ocurre cuando por ejemplo la firm_id que estamos consierando
            # nació después de t_k. En este caso, no tenemos información
            # previa, así que rellenamos con ceros
            return np.zeros((t_k, len(self.feature_cols)), dtype=np.float32)

        return pre_tr_periods

    def _build_train(self) -> list[tuple[np.ndarray, int, int]]:
        """
        Train:
            - Tratados: 1 obs por firma con su cohorte real, label=1
            - NiNis: 1 obs por firma por cohorte, label=0
        """
        records = []

        # Tratados
        for firm_id in self.split['train']['T']:
            firm = self._firms[firm_id]
            cohort_period = int(firm.loc[firm[Col.TRATADO_EN_T], Col.T].iloc[0])
            seq = self._build_sequence(firm_id, cohort_period)
            records.append((seq, cohort_period, 1))

        # NiNis — repetidos por cohorte
        for firm_id in self.split['train']['NiNi']:
            for cohort_period in self._cohorts_periods:
                seq = self._build_sequence(firm_id, cohort_period)
                records.append((seq, cohort_period, 0))

        return records

    def _build_test(self) -> list[tuple[np.ndarray, int, int]]:
        """
        Test:
            - Controles: 1 obs por firma por cohorte. label=1 para su
            cohorte real, label=0 para el resto
            - NiNis: 1 obs por firma por cohorte, label=0
        """
        records = []

        # Controles — repetidos por cohorte
        for firm_id in self.split['test']['C']:
            firm = self._firms[firm_id]
            real_cohort_period = int(firm.loc[firm[Col.CONTROL_EN_T], Col.T].iloc[0])
            for cohort_period in self._cohorts_periods:
                seq   = self._build_sequence(firm_id, cohort_period)
                label = 1 if cohort_period == real_cohort_period else 0
                records.append((seq, cohort_period, label))

        # NiNis — repetidos por cohorte
        for firm_id in self.split['test']['NiNi']:
            for cohort_period in self._cohorts_periods:
                seq = self._build_sequence(firm_id, cohort_period)
                records.append((seq, cohort_period, 0))

        return records

    def _fit_scaler(
        self,
        records: list[tuple[np.ndarray, int, int]]
    ) -> StandardScaler:
        """Fittea el scaler aplanando todas las secuencias de train."""
        flat = np.vstack([seq for seq, _, _ in records])
        scaler = StandardScaler()
        scaler.fit(flat)
        return scaler

    def _scale_records(
        self,
        records: list[tuple[np.ndarray, int, int]]
    ) -> list[tuple[np.ndarray, int, int]]:
        """Aplica el scaler a todas las secuencias."""
        return [
            (self.scaler.transform(seq), cohort, label)
            for seq, cohort, label in records
        ]

    def convert(
        self, fit_scaler: bool = True
    ) -> tuple[PanelSequenceDataset, PanelSequenceDataset]:
        """
        Construye train y test, fitteando el scaler sobre train.

        Returns:
            (train_dataset, test_dataset)
        """
        train_records = self._build_train()
        test_records  = self._build_test()

        if fit_scaler:
            # Fittear scaler solo con secuencias de train
            self.scaler = self._fit_scaler(train_records)

            train_records = self._scale_records(train_records)
            test_records  = self._scale_records(test_records)

        return (
            PanelSequenceDataset(train_records),
            PanelSequenceDataset(test_records),
        )