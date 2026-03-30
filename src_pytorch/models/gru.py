"""
GRU-based Seq2Point model for NILM
Based on: https://dl.acm.org/doi/10.1145/3200947.3201011
"""

import torch
import torch.nn as nn


class GRU_NILM(nn.Module):
    """
    Bidirectional GRU Neural Network for Non-Intrusive Load Monitoring.

    This is a Seq2Point model that takes a window of aggregate power readings
    and predicts the power consumption of a single appliance at the last point.

    Architecture:
        - 1 Conv1D layer
        - 2 Bidirectional GRU layers
        - 2 Dense layers
        - Dropout for regularization
    """

    def __init__(self, input_window_length: int = 199):
        """
        Initialize the GRU model.

        Args:
            input_window_length: Length of the input sequence window (default: 199)
        """
        super().__init__()

        self.input_window_length = input_window_length

        # Conv1D with same padding: output length = input length
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=8,
            kernel_size=4,
            padding='same'
        )
        self.relu = nn.ReLU()

        # Bidirectional GRU layers
        # First GRU: input_size=8 (from conv), hidden_size=32, bidirectional -> output=64
        self.gru1 = nn.GRU(
            input_size=8,
            hidden_size=32,
            batch_first=True,
            bidirectional=True
        )
        self.dropout1 = nn.Dropout(0.5)

        # Second GRU: input_size=64 (from bidirectional GRU1), hidden_size=64
        self.gru2 = nn.GRU(
            input_size=64,  # 32 * 2 from bidirectional
            hidden_size=64,
            batch_first=True,
            bidirectional=True
        )
        self.dropout2 = nn.Dropout(0.5)

        # Dense layers
        # Input: 128 (64 * 2 from bidirectional GRU2)
        self.fc1 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, sequence_length)
               or (batch_size, sequence_length, 1)

        Returns:
            Output tensor of shape (batch_size, 1) - predicted power at last point
        """
        # Handle input shape: (batch, seq_len) -> (batch, 1, seq_len) for Conv1d
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.shape[-1] == 1:
            # (batch, seq_len, 1) -> (batch, 1, seq_len)
            x = x.permute(0, 2, 1)

        # Conv1D: (batch, 1, seq_len) -> (batch, 8, seq_len)
        x = self.conv1(x)
        x = self.relu(x)

        # Reshape for GRU: (batch, 8, seq_len) -> (batch, seq_len, 8)
        x = x.permute(0, 2, 1)

        # GRU layers
        x, _ = self.gru1(x)  # (batch, seq_len, 64)
        x = self.dropout1(x)

        x, _ = self.gru2(x)  # (batch, seq_len, 128)
        x = self.dropout2(x)

        # Take only the last time step output
        x = x[:, -1, :]  # (batch, 128)

        # Dense layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)

        return x


def get_model(input_window_length: int = 199) -> GRU_NILM:
    """Factory function to create a GRU model."""
    return GRU_NILM(input_window_length)
