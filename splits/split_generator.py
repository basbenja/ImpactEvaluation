import json
import numpy as np
import pandas as pd

from data_generation.panel_utils import get_treated_ids, get_control_ids, get_nini_ids


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

        self.split = None

    def _validate_groups(
        self,
        treated_ids: list[int],
        control_ids: list[int],
        nini_ids: list[int],
    ) -> None:
        """Raises if any firm appears in more than one group."""
        T, C, N = set(treated_ids), set(control_ids), set(nini_ids)
        overlaps = {'T∩C': T & C, 'T∩NiNi': T & N, 'C∩NiNi': C & N}
        found = {k: v for k, v in overlaps.items() if v}
        if found:
            raise ValueError(f"Groups not mutually exclusive: {found}")

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
        treated_ids = get_treated_ids(self.panel)
        control_ids = get_control_ids(self.panel)
        nini_ids    = get_nini_ids(self.panel)

        self._validate_groups(treated_ids, control_ids, nini_ids)

        # NiNi shuffleados y partidos
        self.rng.shuffle(nini_ids)
        n_train = int(len(nini_ids) * self.train_nini_ratio)
        train_nini = nini_ids[:n_train]
        test_nini  = nini_ids[n_train:]

        self.split = {
            'train': {'T': treated_ids, 'NiNi': train_nini},
            'test':  {'C': control_ids, 'NiNi': test_nini},
        }

        train_ids = treated_ids + train_nini
        test_ids  = control_ids + test_nini

        return train_ids, test_ids

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