# Copyright (c) 2026 Sotirios Athanasoulias. MIT License — see LICENSE for details.
"""
PyTorch Trainer for NILM models with callbacks support.
"""

import os
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


@dataclass
class TrainingHistory:
    """Stores training history metrics."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_mae: List[float] = field(default_factory=list)
    val_mae: List[float] = field(default_factory=list)
    epochs: List[int] = field(default_factory=list)

    def append(self, epoch: int, train_loss: float, val_loss: float,
               train_mae: float = None, val_mae: float = None):
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        if train_mae is not None:
            self.train_mae.append(train_mae)
        if val_mae is not None:
            self.val_mae.append(val_mae)


class EarlyStopping:
    """
    Early stopping callback to stop training when validation loss stops improving.
    """

    def __init__(
        self,
        patience: int = 6,
        min_delta: float = 1e-6,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss minimization, 'max' for metric maximization
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.

        Args:
            value: Current metric value (e.g., validation loss)

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == 'min':
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print("Early stopping triggered!")

        return self.should_stop

    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_value = None
        self.should_stop = False


class ModelCheckpoint:
    """
    Callback to save model when validation loss improves.
    """

    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        verbose: bool = True
    ):
        """
        Initialize model checkpoint.

        Args:
            filepath: Path to save the model
            monitor: Metric to monitor
            mode: 'min' to save when metric decreases, 'max' for increases
            save_best_only: If True, only save when metric improves
            verbose: Whether to print messages
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose

        self.best_value = None

        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def __call__(self, model: nn.Module, value: float) -> bool:
        """
        Save model if metric improved.

        Args:
            model: PyTorch model to save
            value: Current metric value

        Returns:
            True if model was saved, False otherwise
        """
        should_save = False

        if self.best_value is None:
            should_save = True
        elif self.mode == 'min' and value < self.best_value:
            should_save = True
        elif self.mode == 'max' and value > self.best_value:
            should_save = True

        if should_save or not self.save_best_only:
            if should_save:
                self.best_value = value

            # Save as .pt (primary)
            torch.save(model.state_dict(), self.filepath)

            # Also save as .pth (secondary)
            pth_path = self.filepath.replace('.pt', '.pth')
            torch.save(model.state_dict(), pth_path)

            if self.verbose:
                print(f"Model saved to {self.filepath} and {pth_path} (val_loss: {value:.6f})")
            return True

        return False


class Trainer:
    """
    PyTorch Trainer for NILM models.

    Handles the training loop with support for:
    - Early stopping
    - Model checkpointing
    - TensorBoard logging
    - Learning rate scheduling
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module = None,
        device: str = None,
        cfg = None
    ):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train
            optimizer: Optimizer for training
            loss_fn: Loss function (default: MSELoss)
            device: Device to train on (default: auto-detect)
            cfg: Hydra configuration object (optional)
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn or nn.MSELoss()
        self.cfg = cfg

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Initialize callbacks
        self.early_stopping = None
        self.model_checkpoint = None
        self.tensorboard_writer = None
        self.lr_scheduler = None

        # Training state
        self.history = TrainingHistory()
        self.current_epoch = 0

    def setup_callbacks(
        self,
        checkpoint_dir: str = './checkpoint',
        tensorboard_dir: str = './tensorboard',
        early_stopping_patience: int = 6,
        early_stopping_min_delta: float = 1e-6,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_patience: int = 5,
        lr_scheduler_min_lr: float = 1e-6,
        lr_scheduler_cooldown: int = 2,
    ):
        """
        Setup training callbacks.

        Args:
            checkpoint_dir: Directory for model checkpoints
            tensorboard_dir: Directory for TensorBoard logs
            early_stopping_patience: Patience for early stopping
            early_stopping_min_delta: Min delta for early stopping
            lr_scheduler_factor: Factor by which LR is reduced on plateau
            lr_scheduler_patience: Epochs without val_loss improvement before LR reduction
            lr_scheduler_min_lr: Lower bound on the learning rate
            lr_scheduler_cooldown: Epochs to wait after a reduction before normal operation resumes
        """
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta
        )

        # Model checkpoint
        self.model_checkpoint = ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'model.pt'),
            monitor='val_loss',
            mode='min',
            save_best_only=True
        )

        # TensorBoard
        self.tensorboard_writer = SummaryWriter(log_dir=tensorboard_dir)

        # ReduceLROnPlateau scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            min_lr=lr_scheduler_min_lr,
            cooldown=lr_scheduler_cooldown,
        )

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0

        current_lr = self.optimizer.param_groups[0]['lr']
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)

            # Ensure shapes match for loss calculation
            if outputs.shape != batch_y.shape:
                if outputs.dim() == 2 and batch_y.dim() == 1:
                    batch_y = batch_y.unsqueeze(1)
                elif outputs.dim() == 1 and batch_y.dim() == 2:
                    outputs = outputs.unsqueeze(1)

            loss = self.loss_fn(outputs, batch_y)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            total_mae += torch.mean(torch.abs(outputs - batch_y)).item()
            num_batches += 1

            pbar.set_postfix({'loss': loss.item(), 'lr': f'{current_lr:.2e}'})

        return {
            'loss': total_loss / num_batches,
            'mae': total_mae / num_batches
        }

    @torch.no_grad()
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0

        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            outputs = self.model(batch_x)

            # Ensure shapes match
            if outputs.shape != batch_y.shape:
                if outputs.dim() == 2 and batch_y.dim() == 1:
                    batch_y = batch_y.unsqueeze(1)
                elif outputs.dim() == 1 and batch_y.dim() == 2:
                    outputs = outputs.unsqueeze(1)

            loss = self.loss_fn(outputs, batch_y)

            total_loss += loss.item()
            total_mae += torch.mean(torch.abs(outputs - batch_y)).item()
            num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'mae': total_mae / num_batches
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        verbose: bool = True
    ) -> TrainingHistory:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            verbose: Whether to print progress

        Returns:
            TrainingHistory object with metrics
        """
        best_val_loss = float('inf')
        best_epoch = 0

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate_epoch(val_loader)

            # Log to history
            self.history.append(
                epoch=epoch + 1,
                train_loss=train_metrics['loss'],
                val_loss=val_metrics['loss'],
                train_mae=train_metrics['mae'],
                val_mae=val_metrics['mae']
            )

            # LR scheduler step (must happen before logging so the new LR is recorded)
            if self.lr_scheduler:
                prev_lr = self.optimizer.param_groups[0]['lr']
                self.lr_scheduler.step(val_metrics['loss'])
                current_lr = self.optimizer.param_groups[0]['lr']
                if current_lr < prev_lr and verbose:
                    print(f"ReduceLROnPlateau: LR reduced from {prev_lr:.2e} to {current_lr:.2e}")
            else:
                current_lr = self.optimizer.param_groups[0]['lr']

            # TensorBoard logging
            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
                self.tensorboard_writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
                self.tensorboard_writer.add_scalar('MAE/train', train_metrics['mae'], epoch)
                self.tensorboard_writer.add_scalar('MAE/val', val_metrics['mae'], epoch)
                self.tensorboard_writer.add_scalar('LR', current_lr, epoch)

            # Track best
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_epoch = epoch + 1

            # Print progress
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"loss: {train_metrics['loss']:.4f} - "
                      f"val_loss: {val_metrics['loss']:.4f} - "
                      f"mae: {train_metrics['mae']:.4f} - "
                      f"val_mae: {val_metrics['mae']:.4f} - "
                      f"lr: {current_lr:.2e}")

            # Model checkpoint
            if self.model_checkpoint:
                self.model_checkpoint(self.model, val_metrics['loss'])

            # Early stopping
            if self.early_stopping:
                if self.early_stopping(val_metrics['loss']):
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

        print(f"\nBest epoch: {best_epoch} | Best val_loss: {best_val_loss:.6f}")

        # Save the fully-trained (last-epoch) model alongside the best checkpoint
        if self.model_checkpoint:
            ckpt_dir = os.path.dirname(self.model_checkpoint.filepath)
            final_pt  = os.path.join(ckpt_dir, 'model_final.pt')
            final_pth = os.path.join(ckpt_dir, 'model_final.pth')
            torch.save(self.model.state_dict(), final_pt)
            torch.save(self.model.state_dict(), final_pth)
            print(f"Final model saved to {final_pt} and {final_pth}")

        # Close TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.close()

        return self.history

    def load_checkpoint(self, filepath: str):
        """Load model from checkpoint."""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Model loaded from {filepath}")


def train_model(
    cfg,
    model: nn.Module,
    data,
    optimizer_params: Dict = None
) -> TrainingHistory:
    """
    Train a NILM model using configuration.

    Args:
        cfg: Configuration object (dict-like with nested attributes)
        model: PyTorch model to train
        data: Data loader manager (with .train and .val properties)
        optimizer_params: Override optimizer parameters

    Returns:
        TrainingHistory object
    """
    # Setup optimizer
    if optimizer_params is None:
        optimizer_params = dict(cfg.training.optimizer)

    optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)

    # Setup loss function
    loss_fn = nn.MSELoss()

    # Create trainer
    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, cfg=cfg)

    # Setup callbacks
    trainer.setup_callbacks(
        checkpoint_dir=os.path.join(os.getcwd(), 'checkpoint'),
        tensorboard_dir=os.path.join(os.getcwd(), 'tensorboard'),
        early_stopping_patience=cfg.callbacks.early_stopping.patience,
        early_stopping_min_delta=cfg.callbacks.early_stopping.min_delta
    )

    # Train
    history = trainer.fit(
        train_loader=data.train,
        val_loader=data.val,
        epochs=cfg.training.epochs,
        verbose=True
    )

    return history
