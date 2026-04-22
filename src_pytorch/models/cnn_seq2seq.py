# Copyright (c) 2026 Sotirios Athanasoulias. MIT License — see LICENSE for details.
"""
CNN-based Seq2Seq model for NILM.
Identical architecture to CNN_NILM (Seq2Point) except the final Dense
layer outputs 299 values instead of 1 — predicting the full appliance
power sequence for the input window.
Based on: https://arxiv.org/pdf/1612.09106
"""

import torch
import torch.nn as nn


class CNN_NILM_Seq2Seq(nn.Module):
    """
    Convolutional Neural Network Seq2Seq model for NILM.

    Architecture (identical to CNN_NILM Seq2Point except final output size):
        - 5 × Conv1d layers with ReLU  (no padding, 299 → 270)
        - Flatten
        - Dense(13500, 1024) + ReLU + Dropout
        - Dense(1024, 299)              ← outputs full sequence

    Input  shape: (batch, 299)  or  (batch, 299, 1)
    Output shape: (batch, 299, 1)
    """

    def __init__(self, input_window_length: int = 299):
        """
        Args:
            input_window_length: Length of the input sequence window (default: 299)
        """
        super().__init__()

        self.input_window_length = input_window_length

        # After 5 conv layers (no padding):
        # kernel 10 → L-9, kernel 8 → L-16, kernel 6 → L-21,
        # kernel 5 → L-25, kernel 5 → L-29  →  299-29 = 270
        conv_output_length = input_window_length - 29

        self.network = nn.Sequential(
            nn.Conv1d(in_channels=1,  out_channels=30, kernel_size=10),
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
            nn.Linear(1024, input_window_length),   # 1 → 299
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialisation for Conv1d and Linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) or (batch, seq_len, 1)

        Returns:
            (batch, seq_len, 1)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)                   # (batch, seq_len) → (batch, 1, seq_len)
        elif x.dim() == 3 and x.shape[-1] == 1:
            x = x.permute(0, 2, 1)               # (batch, seq_len, 1) → (batch, 1, seq_len)

        out = self.network(x)                     # (batch, 299)
        return out.unsqueeze(-1)                  # (batch, 299, 1)


def get_model(input_window_length: int = 299) -> CNN_NILM_Seq2Seq:
    """Factory function to create a CNN Seq2Seq model."""
    return CNN_NILM_Seq2Seq(input_window_length)
