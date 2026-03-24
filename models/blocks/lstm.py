import torch
import torch.nn as nn

class LSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional=False):
        """
        input_size: amount of features of each time series
        hidden_size: amount of neurons per layer
        num_layers: amount of LSTM layers
        output_size: amount of classes to classify (2 if we are use CrossEntropyLoss, 
        1 if we use BCEWithLogitsLoss)
        dropout: dropout rate
        """
        super(LSTMBlock, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.D = 1 if not bidirectional else 2

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.D * self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.D * self.num_layers, batch_size, self.hidden_size).to(x.device)

        # output.shape = [batch_size, seq_len, num_directions * hidden_size]
        output, (_, _) = self.lstm(x, (h0, c0))
        lstm_out = output[:, -1, :]
        return lstm_out