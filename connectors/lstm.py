import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from typing import Optional

from connectors.base import BaseConnector, Record
from data_generation.panel_schema import Col
from datasets.panel_sequence import PanelSequenceDataset


class LSTMConnector(BaseConnector):
    """
    Conector que transforma el panel + split en Datasets de PyTorch.

    Args:
        panel        : DataFrame en formato long
        split        : dict generado por SplitGenerator
        feature_cols : columnas a usar como features (int/float, bool, o categorical)

    Encoding:
        - int/float  → StandardScaler
        - bool       → cast a float32, sin escalar
        - categorical (object) → one-hot (pd.get_dummies), sin escalar
    """
    def __init__(
        self,
        panel: pd.DataFrame,
        split: dict,
        feature_cols: list[str]
    ):
        self.split = split
        self.scaler: Optional[StandardScaler] = None

        preprocessed = self.preprocess_panel(panel, feature_cols)
        self.panel       = preprocessed.panel
        self.feature_cols = preprocessed.feature_cols
        self._n_numeric  = preprocessed.n_numeric

        # Índice para acceso rápido por firma
        self._firms = {
            fid: df for fid, df in self.panel.groupby(Col.ID_FIRMA)
        }

        self._cohorts_periods = sorted(
            self.panel.loc[self.panel[Col.TRATADO_EN_T], Col.T].unique().tolist()
        )
        self._period_to_cohort_id = {
            period: idx for idx, period in enumerate(self._cohorts_periods)
        }

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

    def build_train_records(self) -> list[Record]:
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
            cohort_id = self._period_to_cohort_id[cohort_period]
            seq = self._build_sequence(firm_id, cohort_period)
            records.append((firm_id, seq, cohort_id, 1))

        # NiNis — repetidos por cohorte
        for firm_id in self.split['train']['NiNi']:
            for cohort_id, cohort_period in enumerate(self._cohorts_periods):
                seq = self._build_sequence(firm_id, cohort_period)
                records.append((firm_id, seq, cohort_id, 0))

        return records

    def build_test_records(self) -> list[Record]:
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
            real_cohort_id = self._period_to_cohort_id[real_cohort_period]
            for cohort_id, cohort_period in enumerate(self._cohorts_periods):
                seq   = self._build_sequence(firm_id, cohort_period)
                label = 1 if cohort_id == real_cohort_id else 0
                records.append((firm_id, seq, cohort_id, label))

        # NiNis — repetidos por cohorte
        for firm_id in self.split['test']['NiNi']:
            for cohort_id, cohort_period in enumerate(self._cohorts_periods):
                seq = self._build_sequence(firm_id, cohort_period)
                records.append((firm_id, seq, cohort_id, 0))

        return records

    def fit_scaler(
        self,
        records: list[Record]
    ) -> StandardScaler:
        """Fittea el scaler solo sobre las columnas numéricas."""
        flat = np.vstack([seq[:, :self._n_numeric] for _, seq, _, _ in records])
        scaler = StandardScaler()
        scaler.fit(flat)
        return scaler

    def scale_records(
        self,
        records: list[Record],
        scaler: StandardScaler,
    ) -> list[Record]:
        """Escala columnas numéricas; indicadoras y dummies pasan sin cambios."""
        scaled = []
        for firm_id, seq, cohort, label in records:
            numeric  = scaler.transform(seq[:, :self._n_numeric])
            indicators = seq[:, self._n_numeric:]
            scaled.append((firm_id, np.concatenate([numeric, indicators], axis=1), cohort, label))
        return scaled

    def convert(
        self, fit_scaler: bool = True
    ) -> tuple[PanelSequenceDataset, PanelSequenceDataset]:
        """
        Construye train y test, fitteando el scaler sobre train.

        Returns:
            (train_dataset, test_dataset)
        """
        train_records = self.build_train_records()
        test_records  = self.build_test_records()

        if fit_scaler:
            self.scaler = self.fit_scaler(train_records)
            train_records = self.scale_records(train_records, self.scaler)
            test_records  = self.scale_records(test_records, self.scaler)

        return (
            PanelSequenceDataset(train_records),
            PanelSequenceDataset(test_records),
        )
