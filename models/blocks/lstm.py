import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        """
        input_size:  amount of features of each time series
        hidden_size: amount of neurons per layer
        num_layers:  amount of LSTM layers
        output_size: amount of classes to classify (2 if we are use CrossEntropyLoss,
            1 if we use BCEWithLogitsLoss)
        dropout: dropout rate
        """
        super(LSTMBlock, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, n_features).
                Secuencia de características con el padding aplicado para poder
                manejar secuencias de longitud variable.
            lengths: (batch_size,).
                Longitudes de las secuencias originales (antes de hacer el
                padding).

        Returns:
            logits:(batch_size, 1)
        """
        # Empaquetamos la secuencia para que el LSTM no procese los pasos de tiempo
        # que corresponden al padding
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # output.shape = [batch_size, seq_len, num_directions * hidden_size]
        output, (hn, cn) = self.lstm(packed)

        # Desempaquetamos la secuencia para volver a tener una forma de tensor regular
        # output.shape = (batch_size, max_seq_len, hidden_size)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # lengths-1 (es un vector) porque las posiciones se indexan desde 0, así obtenemos
        # la salida del último paso de tiempo real para cada secuencia
        lstm_out = output[range(len(lengths)), lengths-1, :]
        return lstm_out