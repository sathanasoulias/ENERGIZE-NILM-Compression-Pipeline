"""
CNN-based Seq2Point model for NILM
Based on: https://arxiv.org/pdf/1612.09106
"""

import torch
import torch.nn as nn


class CNN_NILM(nn.Module):
    """
    Convolutional Neural Network for Non-Intrusive Load Monitoring.

    This is a Seq2Point model that takes a window of aggregate power readings
    and predicts the power consumption of a single appliance at the center point.

    Architecture:
        - 5 Conv1D layers with ReLU activation
        - 2 Dense layers
        - Dropout for regularization
    """

    def __init__(self, input_window_length: int = 299):
        """
        Initialize the CNN model.

        Args:
            input_window_length: Length of the input sequence window (default: 299)
        """
        super().__init__()

        self.input_window_length = input_window_length

        # Calculate output sizes after each conv layer (no padding, stride=1)
        # After conv1 (kernel=10): L - 10 + 1 = L - 9
        # After conv2 (kernel=8): (L-9) - 8 + 1 = L - 16
        # After conv3 (kernel=6): (L-16) - 6 + 1 = L - 21
        # After conv4 (kernel=5): (L-21) - 5 + 1 = L - 25
        # After conv5 (kernel=5): (L-25) - 5 + 1 = L - 29

        conv_output_length = input_window_length - 29  # 299 - 29 = 270

        self.network = nn.Sequential(
            # Conv layers expect (batch, channels, length)
            nn.Conv1d(in_channels=1, out_channels=30, kernel_size=10),
            nn.ReLU(),
            nn.Conv1d(in_channels=30, out_channels=30, kernel_size=8),
            nn.ReLU(),
            nn.Conv1d(in_channels=30, out_channels=40, kernel_size=6),
            nn.ReLU(),
            nn.Conv1d(in_channels=40, out_channels=50, kernel_size=5),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=50, out_channels=50, kernel_size=5),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(50 * conv_output_length, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Glorot/Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, sequence_length)
               or (batch_size, sequence_length, 1)

        Returns:
            Output tensor of shape (batch_size, 1) - predicted power at center point
        """
        # Handle input shape: (batch, seq_len) -> (batch, 1, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.shape[-1] == 1:
            # (batch, seq_len, 1) -> (batch, 1, seq_len)
            x = x.permute(0, 2, 1)

        return self.network(x)


def get_model(input_window_length: int = 299) -> CNN_NILM:
    """Factory function to create a CNN model."""
    return CNN_NILM(input_window_length)
