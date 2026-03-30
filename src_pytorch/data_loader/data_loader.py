"""
PyTorch Data Loader for NILM experiments.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from numpy.lib.stride_tricks import sliding_window_view
from typing import Tuple


class NILMDataset(Dataset):
    """
    PyTorch Dataset for NILM (Non-Intrusive Load Monitoring).

    Handles windowing strategies for different model architectures:
    - CNN/GRU (Seq2Point): Sliding windows with single target point
    - TCN (Seq2Seq): Non-overlapping windows with sequence targets
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        input_window_length: int,
        model_name: str,
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize the NILM Dataset.

        Args:
            data: Input aggregate power data (1D array)
            labels: Target appliance power data (1D array)
            input_window_length: Length of input sequence window
            model_name: Name of model ('cnn', 'gru', 'tcn')
            dtype: PyTorch data type for tensors
        """
        self.input_window_length = input_window_length
        self.model_name = model_name.lower()
        self.dtype = dtype

        if self.model_name in ['cnn', 'gru']:
            self._prepare_seq2point(data, labels)
        elif self.model_name == 'tcn':
            self._prepare_seq2seq(data, labels)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def _prepare_seq2point(self, data: np.ndarray, labels: np.ndarray):
        """Prepare data for Seq2Point models (CNN, GRU)."""
        # Create sliding windows for inputs
        self.inputs = sliding_window_view(data, self.input_window_length)

        # Determine target offset based on model
        if self.model_name == 'cnn':
            # Target is center point of window
            offset = self.input_window_length // 2
            self.targets = labels[offset:offset + len(self.inputs)]
        else:  # gru
            # Target is last point of window
            offset = self.input_window_length - 1
            self.targets = labels[offset:offset + len(self.inputs)]

        # Ensure matching lengths
        min_len = min(len(self.inputs), len(self.targets))
        self.inputs = self.inputs[:min_len].astype(np.float32)
        self.targets = self.targets[:min_len].astype(np.float32)

    def _prepare_seq2seq(self, data: np.ndarray, labels: np.ndarray):
        """Prepare data for Seq2Seq models (TCN)."""
        # Non-overlapping windows
        self.inputs = sliding_window_view(
            data, self.input_window_length
        )[::self.input_window_length, :]

        self.targets = sliding_window_view(
            labels, self.input_window_length
        )[::self.input_window_length, :]

        # Ensure matching lengths
        min_len = min(len(self.inputs), len(self.targets))
        self.inputs = self.inputs[:min_len].astype(np.float32)
        self.targets = self.targets[:min_len].astype(np.float32)

        # Add channel dimension: (batch, window_length) -> (batch, window_length, 1)
        self.inputs = np.expand_dims(self.inputs, axis=-1)
        self.targets = np.expand_dims(self.targets, axis=-1)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.inputs[idx], dtype=self.dtype)
        y = torch.tensor(self.targets[idx], dtype=self.dtype)

        # Ensure target shape
        if self.model_name in ['cnn', 'gru'] and y.dim() == 0:
            y = y.unsqueeze(0)

        return x, y


class SimpleNILMDataLoader:
    """
    Data loader for NILM experiments.
    Loads CSV data and creates PyTorch DataLoaders for training, validation, and testing.
    """

    def __init__(
        self,
        data_dir: str,
        model_name: str,
        batch_size: int,
        input_window_length: int,
        train: bool = True,
        num_workers: int = 0
    ):
        """
        Initialize the data loader.

        Args:
            data_dir: Path to directory containing CSV files
            model_name: Name of model ('cnn', 'gru', 'tcn')
            batch_size: Batch size for DataLoader
            input_window_length: Length of input sequence window
            train: If True, load train/val/test; else only test
            num_workers: Number of DataLoader workers
        """
        self.model_name = model_name.lower()
        self.batch_size = batch_size
        self.input_window_length = input_window_length
        self.num_workers = num_workers

        if train:
            train_data = np.array(pd.read_csv(os.path.join(data_dir, 'training_.csv')))
            val_data = np.array(pd.read_csv(os.path.join(data_dir, 'validation_.csv')))
            test_data = np.array(pd.read_csv(os.path.join(data_dir, 'test_.csv')))

            self.train_data = train_data[:, 0]
            self.train_labels = train_data[:, 1]

            self.val_data = val_data[:, 0]
            self.val_labels = val_data[:, 1]

            self.test_data = test_data[:, 0]
            self.test_labels = test_data[:, 1]
        else:
            test_data = np.array(pd.read_csv(os.path.join(data_dir, 'test_.csv')))
            self.test_data = test_data[:, 0]
            self.test_labels = test_data[:, 1]

    def _create_dataloader(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        shuffle: bool = False,
        drop_last: bool = False
    ) -> DataLoader:
        """Create a DataLoader from numpy arrays."""
        dataset = NILMDataset(
            data=data,
            labels=labels,
            input_window_length=self.input_window_length,
            model_name=self.model_name
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=drop_last,
            pin_memory=torch.cuda.is_available()
        )

    @property
    def train(self) -> DataLoader:
        """Returns the training DataLoader (shuffled)."""
        return self._create_dataloader(
            self.train_data, self.train_labels,
            shuffle=True, drop_last=True
        )

    @property
    def val(self) -> DataLoader:
        """Returns the validation DataLoader."""
        return self._create_dataloader(
            self.val_data, self.val_labels,
            shuffle=False, drop_last=False
        )

    @property
    def test(self) -> DataLoader:
        """Returns the test DataLoader."""
        return self._create_dataloader(
            self.test_data, self.test_labels,
            shuffle=False, drop_last=False
        )


# Alias for backwards compatibility
DataLoaderNILM = SimpleNILMDataLoader
