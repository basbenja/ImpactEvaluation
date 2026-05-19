import numpy as np
import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


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

    def __init__(self, records: list[tuple[int, np.ndarray, int, int]]):
        """
        Args:
            records: lista de (firm_id, secuencia, cohorte, label)
        """
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return PanelSequenceDataset(self.records[idx])

        firm_id, seq, cohort, label = self.records[idx]
        return (
            torch.tensor(firm_id, dtype=torch.long),
            torch.tensor(seq,     dtype=torch.float32),
            torch.tensor(cohort,  dtype=torch.long),
            torch.tensor(label,   dtype=torch.float32),
        )

    @staticmethod
    def collate_fn(batch):
        """
        batch: lista de (sequence, cohort, label)
        """
        firm_ids, sequences, cohorts, labels = zip(*batch)

        # Tenemos que devolver los largo originales para que el modelo sepa hasta
        # dónde leer (esto después se le pasa a pack_padded_sequence)
        lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long)

        # pad_sequence apila y rellena con 0s hasta la longitud máxima del batch
        # sequences_padded: (batch_size, T_max, F)
        sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)

        return (
            torch.stack(firm_ids),
            sequences_padded,
            lengths,
            torch.stack(cohorts),
            torch.stack(labels),
        )
