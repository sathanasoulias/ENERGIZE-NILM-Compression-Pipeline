"""
Utility functions for PyTorch NILM training.
"""

import os
import random
from typing import List, Optional

import numpy as np
import torch


def set_seeds(seed: int = 6):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For reproducibility (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Seeds set to {seed}")


def create_experiment_directories(root_dir: str, directories: List[str]) -> None:
    """
    Create directories for experiment outputs.

    Args:
        root_dir: Root directory path
        directories: List of subdirectory names to create
    """
    for directory in directories:
        dir_path = os.path.join(root_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for training.

    Args:
        device: Specific device string (e.g., 'cuda:0', 'cpu')

    Returns:
        torch.device object
    """
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")

    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: torch.nn.Module, input_shape: tuple):
    """
    Print a summary of the model architecture.

    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (without batch dimension)
    """
    print("=" * 60)
    print("Model Summary")
    print("=" * 60)
    print(model)
    print("-" * 60)
    print(f"Total trainable parameters: {count_parameters(model):,}")
    print("=" * 60)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str
):
    """
    Save a full training checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> dict:
    """
    Load a training checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to map checkpoint to

    Returns:
        Dictionary with checkpoint info (epoch, loss)
    """
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Checkpoint loaded: {filepath}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A')}")

    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf'))
    }


def value_checks(cfg) -> None:
    """
    Validate configuration values.

    Args:
        cfg: Hydra configuration object
    """
    if cfg.model.name == 'seq2subseq' or cfg.model.name == 'seq2seq':
        if cfg.model.init.input_window_length % 2 != 0:
            raise ValueError('Input width must be divisible by 2 for seq2subseq model')

    if cfg.model.name == 'cnn':
        if cfg.model.init.input_window_length % 2 != 1:
            raise ValueError('Input width must be an odd number for seq2point (CNN) model')


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']
