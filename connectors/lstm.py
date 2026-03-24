import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from typing import Optional


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
        seq, cohort, label = self.records[idx]
        return (
            torch.tensor(seq,    dtype=torch.float32),
            torch.tensor(cohort, dtype=torch.long),
            torch.tensor(label,  dtype=torch.float32),
        )


class LSTMConnector:
    """
    Conector que transforma el panel + split en Datasets de PyTorch.

    Args:
        panel        : DataFrame en formato long
        split        : dict generado por SplitGenerator
        feature_cols : columnas a usar como features; si es None se
            detectan automáticamente
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        split: dict,
        feature_cols: list[str]
    ):
        self.panel = panel.sort_values(['firm_id', 't']).reset_index(drop=True)
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
            fid: df for fid, df in self.panel.groupby('firm_id')
        }

        # Período de inicio de cada cohorte: {cohort_id: t_k}
        self._cohort_periods = self._get_cohort_periods()

        # Lista de todas las cohortes
        self._cohorts = sorted(self._cohort_periods.keys())

    def _get_cohort_periods(self) -> dict[int, int]:
        """
        Infiere el período de inicio de cada cohorte desde el split.
        t_k es el primer t en que aparece un tratado de esa cohorte
        con label treated=True.
        """
        treated = self.panel[self.panel['treated']]
        return (
            treated.groupby('cohort')['cohort_start']
            .min()
            .astype(int)
            .to_dict()
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
            # nació después de t_k. En este caso, no tenemos información
            # previa, así que rellenamos con ceros
            return np.zeros((t_k, len(self.feature_cols)), dtype=np.float32)

        # Si hay menos períodos que t_k, hacemos padding hacia atrás con el
        # primer valor conocido
        # NOTE: para esto hay que ver qué hacemos, la idea sería no agregar
        # valores ficticios, sino usar los que están disponibles. Por ejemplo,
        # si t_k=5 pero solo hay datos hasta t=3, la secuencia sería de largo 3,
        # no 5. Esto es importante para que el modelo aprenda a manejar
        # secuencias de largo variable.
        if len(pre_tr_periods) < t_k:
            pad = np.tile(pre_tr_periods[0], (t_k - len(pre_tr_periods), 1))
            pre_tr_periods = np.vstack([pad, pre_tr_periods])

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
            cohort_id = firm['cohort'].iloc[0]
            t_k = self._cohort_periods[cohort_id]
            seq = self._build_sequence(firm_id, t_k)
            records.append((seq, cohort_id, 1))

        # NiNis — repetidos por cohorte
        for firm_id in self.split['train']['NiNi']:
            for cohort_id in self._cohorts:
                t_k = self._cohort_periods[cohort_id]
                seq = self._build_sequence(firm_id, t_k)
                records.append((seq, cohort_id, 0))

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
            real_cohort_id = self._firms[firm_id]['cohort'].iloc[0]
            for cohort_id in self._cohorts:
                t_k   = self._cohort_periods[cohort_id]
                seq   = self._build_sequence(firm_id, t_k)
                label = 1 if cohort_id == real_cohort_id else 0
                records.append((seq, cohort_id, label))

        # NiNis — repetidos por cohorte
        for firm_id in self.split['test']['NiNi']:
            for cohort_id in self._cohorts:
                t_k = self._cohort_periods[cohort_id]
                seq = self._build_sequence(firm_id, t_k)
                records.append((seq, cohort_id, 0))

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