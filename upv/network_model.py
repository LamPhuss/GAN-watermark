"""
network_model.py  (ported from MarkLLM/watermark/upv/network_model.py — standalone)
Description: Neural network definition for the UPV watermark algorithm.
  - UPVSubNet:    Shared MLP sub-network (used by both Generator and Detector)
  - UPVGenerator: Predicts whether a (context, candidate) token pair is "green"
  - UPVDetector:  Classifies a token sequence as watermarked or not (LSTM-based)

No changes to logic — only removed MarkLLM-specific imports.
"""

import torch
from torch import nn


class UPVSubNet(nn.Module):
    """Shared MLP sub-network used by both UPVGenerator and UPVDetector."""

    def __init__(self, input_dim: int, num_layers: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class UPVGenerator(nn.Module):
    """
    Watermark Generator for UPV.
    Input:  (batch, window_size, bit_number) — token window as binary vectors
    Output: (batch, 1) probability that the window is in the "green" list
    """

    def __init__(
        self,
        input_dim: int,
        window_size: int,
        num_layers: int = 5,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.sub_net = UPVSubNet(input_dim, num_layers, hidden_dim)
        self.window_size = window_size
        self.relu = nn.ReLU()
        self.combine_layer = nn.Linear(window_size * hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window_size, input_dim)"""
        batch_size = x.shape[0]
        x = x.view(-1, x.shape[-1])           # (batch*window, input_dim)
        sub_out = self.sub_net(x)              # (batch*window, hidden_dim)
        sub_out = sub_out.view(batch_size, -1) # (batch, window*hidden_dim)
        combined = self.combine_layer(sub_out)
        combined = self.relu(combined)
        out = self.output_layer(combined)
        return self.sigmoid(out)               # (batch, 1)


class UPVDetector(nn.Module):
    """
    Watermark Detector for UPV.
    Input:  (batch, seq_len, bit_number) — token sequence as binary vectors
    Output: (batch, 1) probability that the sequence is watermarked
    """

    def __init__(
        self,
        bit_number: int,
        b_layers: int = 5,
        input_dim: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 1,
        num_layers: int = 2,
    ):
        super().__init__()
        self.binary_classifier = UPVSubNet(bit_number, b_layers)
        self.classifier = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, bit_number)"""
        x1 = x.view(-1, x.shape[-1])          # (batch*seq_len, bit_number)
        features = self.binary_classifier(x1)  # (batch*seq_len, hidden_dim=64)
        features = features.view(x.size(0), x.size(1), -1)  # (batch, seq_len, 64)
        lstm_out, _ = self.classifier(features)
        last_out = lstm_out[:, -1, :]          # (batch, hidden_dim)
        hidden = self.fc_hidden(last_out)
        hidden = self.sigmoid(hidden)
        out = self.fc(hidden)
        return self.sigmoid(out)               # (batch, 1)
