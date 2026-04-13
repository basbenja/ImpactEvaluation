import json
import numpy as np
import pandas as pd


class SplitGenerator:
    """
    Genera y persiste el split train/test a nivel firma.

    Train: todos los T (label=1) + train_nini_ratio de los NiNi (label=0)
    Test:  todos los C (label=1) + el resto de los NiNi (label=0)

    Args:
        panel           : DataFrame del panel en formato long
        train_nini_ratio: fracción de NiNi que va a train
        seed            : semilla para reproducibilidad
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        train_nini_ratio: float = 0.5,
        seed: int = 42,
    ):
        self.panel = panel
        self.train_nini_ratio = train_nini_ratio
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self._status = self._get_firm_status()
        self.split = None

    def _get_firm_status(self) -> pd.DataFrame:
        """
        Determina el status definitivo de cada firma usando el último período.
        """
        last = self.panel.sort_values('t').groupby('id_firma').last()
        status = last[['tratado', 'control']].reset_index()
        status['is_T']    = status['tratado']
        status['is_C']    = ~status['tratado'] & status['control']
        status['is_NiNi'] = ~status['tratado'] & ~status['control']
        return status

    def _get_group_ids(self, group: str) -> list:
        """
        Obtiene los firm_ids de T o C.

        Returns:
            [id1, id2, ...]
        """
        if group == 'T':
            mask = self._status['is_T']
        elif group == 'C':
            mask = self._status['is_C']
        elif group == 'NiNi':
            mask = self._status['is_NiNi']

        firms_ids = self._status[mask]['id_firma'].values
        return [int(fid) for fid in firms_ids]

    def generate(self) -> dict:
        """
        Genera el split y lo almacena en self.split.

        Returns:
            dict con la estructura:
            {
                "train": {
                    "T": [...], "NiNi": [...]
                },
                "test": {
                    "C": [...], "NiNi": [...]
                },
                "meta": {...}
            }
        """
        treated_ids = self._get_group_ids(group='T')
        control_ids = self._get_group_ids(group='C')
        nini_ids    = self._get_group_ids(group='NiNi')

        # NiNi shuffleados y partidos
        self.rng.shuffle(nini_ids)
        n_train = int(len(nini_ids) * self.train_nini_ratio)
        train_nini = nini_ids[:n_train]
        test_nini  = nini_ids[n_train:]

        self.split = {
            'train': {'T': treated_ids, 'NiNi': train_nini},
            'test':  {'C': control_ids, 'NiNi': test_nini},
        }

        train = treated_ids + train_nini
        test  = control_ids + test_nini

        return train, test

    def save(self, path: str):
        """Guarda el split como JSON."""
        if self.split is None:
            raise ValueError("Llamá a generate() antes de save().")

        with open(path, 'w') as f:
            json.dump(self.split, f, indent=2)

        print(f"Split guardado en {path}")
        self._print_summary()

    @classmethod
    def load(cls, path: str) -> dict:
        """Carga un split previamente guardado."""
        with open(path) as f:
            return json.load(f)