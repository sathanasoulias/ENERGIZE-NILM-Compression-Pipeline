"""
TCN (Temporal Convolutional Network) based Seq2Seq model for NILM
Based on: https://arxiv.org/pdf/1902.08736 (WaveNet-style architecture)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import numpy as np


class CausalConv1d(nn.Module):
    """
    Causal 1D Convolution that ensures no future information leaks into predictions.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int = 1, bias: bool = True):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        # Remove the extra padding from the end to make it causal
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class GatedBlock(nn.Module):
    """
    Gated block with signal and gate convolutions (WaveNet-style).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float = 0.1, l2_reg: float = 0.0):
        super().__init__()

        # Signal path
        self.signal_conv = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )

        # Gate path
        self.gate_conv = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Apply L2 regularization through weight decay in optimizer
        self.l2_reg = l2_reg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Signal path with ReLU
        signal = F.relu(self.signal_conv(x))

        # Gate path with Sigmoid
        gate = torch.sigmoid(self.gate_conv(x))

        # Gated output
        gated = signal * gate

        # Dropout
        out = self.dropout(gated)

        return out


class TCN_NILM(nn.Module):
    """
    Temporal Convolutional Network for Non-Intrusive Load Monitoring.

    This is a Seq2Seq model based on WaveNet architecture that takes a window
    of aggregate power readings and predicts the power consumption sequence.

    Architecture:
        - Initial 1x1 convolution for feature mixing
        - Multiple stacks of dilated causal convolutions with gating
        - Skip connections concatenated
        - Final dense layer with LeakyReLU
    """

    def __init__(
        self,
        input_window_length: int = 600,
        depth: int = 9,
        nb_filters: Optional[List[int]] = None,
        res_l2: float = 0.0,
        stacks: int = 1,
        dropout: float = 0.1
    ):
        """
        Initialize the TCN model.

        Args:
            input_window_length: Length of the input sequence window
            depth: Number of dilated conv layers per stack
            nb_filters: List of filter counts for each depth level
            res_l2: L2 regularization factor (applied via weight decay)
            stacks: Number of stacking iterations
            dropout: Dropout rate
        """
        super().__init__()

        self.input_window_length = input_window_length
        self.depth = depth
        self.stacks = stacks
        self.res_l2 = res_l2

        # Default filter configuration
        if nb_filters is None:
            nb_filters = [512, 256, 256, 128, 128, 256, 256, 256, 512]

        # Expand to depth if only one value provided
        if len(nb_filters) == 1:
            nb_filters = [nb_filters[0]] * depth

        self.nb_filters = nb_filters

        # Initial 1x1 convolution for feature mixing
        self.initial_conv = nn.Conv1d(1, nb_filters[0], kernel_size=1, padding=0)

        # Create gated blocks for each stack and depth
        self.gated_blocks = nn.ModuleList()

        for _ in range(stacks):
            for i in range(depth):
                in_channels = nb_filters[i - 1] if i > 0 else nb_filters[0]
                out_channels = nb_filters[i]
                dilation = 2 ** i

                block = GatedBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2,
                    dilation=dilation,
                    dropout=dropout,
                    l2_reg=res_l2
                )
                self.gated_blocks.append(block)

        # Calculate total skip connection channels
        # Initial conv output + all gated block outputs
        total_skip_channels = nb_filters[0] + sum(nb_filters) * stacks

        # Final layers (TimeDistributed Dense equivalent)
        self.final_conv = nn.Conv1d(total_skip_channels, 1, kernel_size=1)
        self.leaky_relu = nn.LeakyReLU(0.1)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, 1)
               or (batch_size, 1, sequence_length)

        Returns:
            Output tensor of shape (batch_size, sequence_length, 1)
        """
        # Handle input shape for Conv1d: need (batch, channels, length)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, seq_len) -> (batch, 1, seq_len)
        elif x.dim() == 3:
            if x.shape[-1] == 1:
                # (batch, seq_len, 1) -> (batch, 1, seq_len)
                x = x.permute(0, 2, 1)
            # else: already (batch, 1, seq_len) or (batch, channels, seq_len)

        # Initial feature mixing
        out = self.initial_conv(x)

        # Collect skip connections
        skip_connections = [out]

        # Process through gated blocks
        for block in self.gated_blocks:
            out = block(out)
            skip_connections.append(out)

        # Concatenate all skip connections
        out = torch.cat(skip_connections, dim=1)

        # Final projection
        out = self.final_conv(out)
        out = self.leaky_relu(out)

        # Return shape: (batch, seq_len, 1)
        out = out.permute(0, 2, 1)

        return out


def get_model(
    input_window_length: int = 600,
    depth: int = 9,
    nb_filters: Optional[List[int]] = None,
    res_l2: float = 0.0,
    stacks: int = 1,
    dropout: float = 0.1
) -> TCN_NILM:
    """Factory function to create a TCN model."""
    return TCN_NILM(
        input_window_length=input_window_length,
        depth=depth,
        nb_filters=nb_filters,
        res_l2=res_l2,
        stacks=stacks,
        dropout=dropout
    )
