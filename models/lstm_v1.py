import torch.nn as nn

from models.blocks.lstm import LSTMBlock
from models.blocks.dense import DenseBlock

class LSTMClassifier_v1(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, dropout):
        super().__init__()

        self.lstm = LSTMBlock(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        self.fc = DenseBlock(
            input_size=hidden_size,
            hidden_sizes=[128],
            dropout=dropout
        )

    def forward(self, x):
        lstm_out = self.lstm(x)
        logits = self.fc(lstm_out)
        return logits
