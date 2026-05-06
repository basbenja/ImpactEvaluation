import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks.lstm import LSTMBlock
from models.blocks.dense import DenseBlock

class LSTMClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        n_cohorts: int,
        dropout: float,
    ):
        super().__init__()

        self.n_cohorts = n_cohorts

        self.lstm = LSTMBlock(
            input_size=n_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout
        )
        self.fc = DenseBlock(
            input_size=lstm_hidden_size + n_cohorts,
            hidden_sizes=[128],
            dropout=dropout
        )

    def forward(
        self,
        firm_ids: torch.Tensor,
        x_seq: torch.Tensor,
        lengths: torch.Tensor,
        cohorts: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            firm_ids: (batch_size,).
                Enteros con el id de la firma. No se usan en el modelo pero se
                incluyen para poder trackear los resultados por firma.
            x_seq: (batch_size, seq_len, n_features).
                Secuencia de características con el padding aplicado para poder
                manejar secuencias de longitud variable.
            lengths: (batch_size,).
                Longitudes de las secuencias originales (antes de hacer el
                padding).
            cohorts: (batch_size,).
                Enteros con el id de la cohorte.

        Returns:
            logits:(batch_size, 1)
        """
        lstm_out = self.lstm(x_seq, lengths)

        cohorts_oneshot = F.one_hot(cohorts, num_classes=self.n_cohorts).float()
        combined = torch.cat([lstm_out, cohorts_oneshot], dim=1)

        logits = self.fc(combined)

        return logits
