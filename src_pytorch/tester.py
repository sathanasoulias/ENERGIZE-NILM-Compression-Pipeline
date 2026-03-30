"""
PyTorch Tester for NILM model evaluation.
"""

import os
import csv
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, confusion_matrix
from tqdm import tqdm


def compute_step_function(data: np.ndarray, threshold: float) -> np.ndarray:
    """
    Convert continuous power values to binary ON/OFF states.

    Args:
        data: Power values array
        threshold: Threshold for ON/OFF classification

    Returns:
        Binary array (1 = ON, 0 = OFF)
    """
    return (data >= threshold).astype(int)


def acc_precision_recall_f1_score(
    status: np.ndarray,
    status_pred: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Calculate classification metrics.

    Args:
        status: Ground truth binary status
        status_pred: Predicted binary status

    Returns:
        Tuple of (F1 score, accuracy, precision, recall)
    """
    assert status.shape == status_pred.shape

    tn, fp, fn, tp = confusion_matrix(status, status_pred, labels=[0, 1]).ravel()

    acc = (tn + tp) / (tn + fp + fn + tp)
    precision = tp / max(tp + fp, 1e-9)
    recall = tp / max(tp + fn, 1e-9)
    f1_score = 2 * (precision * recall) / max(precision + recall, 1e-9)

    return f1_score, acc, precision, recall


class Tester:
    """
    Tester class for evaluating NILM models.

    Handles:
    - Model inference
    - Metrics calculation (MAE, F1, Precision, Recall, Accuracy)
    - Results saving to CSV
    """

    def __init__(self, cfg, transfer_test: bool = False):
        """
        Initialize the tester.

        Args:
            cfg: Configuration object (dict-like with nested attributes)
            transfer_test: Whether this is a transfer learning test
        """
        self.cfg = cfg
        self.model_name = cfg.model.name
        self.transfer_test = transfer_test
        self.input_window_length = cfg.model.init.input_window_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @torch.no_grad()
    def predict(
        self,
        model: nn.Module,
        data_loader: DataLoader
    ) -> np.ndarray:
        """
        Generate predictions using the model.

        Args:
            model: PyTorch model
            data_loader: Test data loader

        Returns:
            Predictions as numpy array
        """
        model.eval()
        model.to(self.device)

        predictions = []

        for batch_x, _ in tqdm(data_loader, desc="Predicting"):
            batch_x = batch_x.to(self.device)
            outputs = model(batch_x)
            predictions.append(outputs.cpu().numpy())

        return np.concatenate(predictions).flatten()

    def test(
        self,
        model: nn.Module,
        data,
        reduction_test_data: Optional[DataLoader] = None,
        save_results: bool = True
    ) -> Tuple[float, float]:
        """
        Test the model and calculate metrics.

        Args:
            model: PyTorch model to test
            data: Data loader manager with test_data and test_labels
            reduction_test_data: Optional alternative test data loader
            save_results: Whether to save results to CSV

        Returns:
            Tuple of (MAE, F1 score)
        """
        if reduction_test_data:
            predictions = self.predict(model, reduction_test_data)
            self.input_window_length = data.input_window_length
            self.model_name = data.model_name
        else:
            predictions = self.predict(model, data.test)

        ground_truth = data.test_labels
        agg = data.test_data

        mae, f1, precision, recall, acc = self.model_test(
            self.cfg, predictions, ground_truth, agg
        )

        if save_results:
            self.write_results_to_csv(
                mae, f1, precision, recall, acc,
                self.cfg.appliance.name
            )

        return mae, f1

    def model_test(
        self,
        cfg,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        agg: np.ndarray
    ) -> Tuple[float, float, float, float, float]:
        """
        Process predictions and calculate metrics.

        Args:
            cfg: Configuration object
            predictions: Model predictions
            ground_truth: Ground truth values
            agg: Aggregate power values

        Returns:
            Tuple of (MAE, F1, precision, recall, accuracy)
        """
        # Align predictions with ground truth based on model type
        if self.model_name == 'cnn':
            offset = int(self.input_window_length / 2) - 1
            ground_truth = ground_truth[offset:]
            ground_truth = ground_truth[:len(predictions)]
            agg = agg[offset:]
            agg = agg[:len(predictions)]

        elif self.model_name == 'gru':
            offset = self.input_window_length - 1
            agg = agg[offset:]
            ground_truth = ground_truth[offset:]
            ground_truth = ground_truth[:len(predictions)]
            agg = agg[:len(predictions)]

        elif self.model_name == 'tcn':
            ground_truth = ground_truth[:len(predictions)]
            agg = agg[:len(predictions)]

        assert len(ground_truth) == len(predictions), \
            f'GT: {len(ground_truth)} | PRED: {len(predictions)}'

        # Denormalize
        appliance_name = cfg.appliance.name
        agg = (agg * cfg.dataset.aggregate[appliance_name].std +
               cfg.dataset.aggregate[appliance_name].mean)
        ground_truth = ground_truth * cfg.dataset.cutoff[appliance_name]
        predictions = predictions * cfg.dataset.cutoff[appliance_name]

        # Apply threshold and clip
        threshold = cfg.dataset.threshold[appliance_name]
        cutoff = cfg.dataset.cutoff[appliance_name]

        predictions[predictions < threshold] = 0
        predictions[predictions > cutoff] = cutoff

        # Calculate MAE
        mae = mean_absolute_error(ground_truth, predictions)
        print(f'MAE: {round(mae, 4)}')

        # Calculate classification metrics
        gt_step = compute_step_function(ground_truth, threshold)
        pred_step = compute_step_function(predictions, threshold)

        f1, acc, precision, recall = acc_precision_recall_f1_score(gt_step, pred_step)

        print(f'F1: {f1:.4f} | ACC: {acc:.4f}')

        return mae, f1, precision, recall, acc

    def write_results_to_csv(
        self,
        mae: float,
        f1: float,
        precision: float,
        recall: float,
        acc: float,
        app_name: str
    ):
        """
        Save results to CSV file.

        Args:
            mae: Mean Absolute Error
            f1: F1 score
            precision: Precision
            recall: Recall
            acc: Accuracy
            app_name: Appliance name
        """
        os.makedirs(os.path.join(os.getcwd(), 'metrics'), exist_ok=True)
        save_path = os.path.join(os.getcwd(), 'metrics', f'{app_name}_results.csv')

        with open(save_path, 'w', newline='') as csv_file:
            header_key = ['Model', 'MAE', 'F1', 'Precision', 'Recall', 'Acc']
            writer = csv.DictWriter(csv_file, fieldnames=header_key)

            writer.writeheader()
            writer.writerow({
                'Model': self.model_name,
                'MAE': round(mae, 3),
                'F1': round(f1, 3),
                'Precision': round(precision, 3),
                'Recall': round(recall, 3),
                'Acc': round(acc, 3)
            })

        print(f"Results saved to {save_path}")


class SimpleTester:
    """
    Simplified tester for use without Hydra configuration.
    Useful for notebooks and standalone scripts.
    """

    def __init__(
        self,
        model_name: str,
        input_window_length: int,
        threshold: float,
        cutoff: float,
        mean: float = 0.0,
        std: float = 1.0
    ):
        """
        Initialize the simple tester.

        Args:
            model_name: Name of model ('cnn', 'gru', 'tcn')
            input_window_length: Input window length
            threshold: ON/OFF threshold for appliance
            cutoff: Maximum power cutoff value
            mean: Mean for denormalization of aggregate
            std: Std for denormalization of aggregate
        """
        self.model_name = model_name.lower()
        self.input_window_length = input_window_length
        self.threshold = threshold
        self.cutoff = cutoff
        self.mean = mean
        self.std = std
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @torch.no_grad()
    def predict(
        self,
        model: nn.Module,
        data_loader: DataLoader
    ) -> np.ndarray:
        """Generate predictions."""
        model.eval()
        model.to(self.device)

        predictions = []

        for batch_x, _ in tqdm(data_loader, desc="Predicting"):
            batch_x = batch_x.to(self.device)
            outputs = model(batch_x)
            predictions.append(outputs.cpu().numpy())

        return np.concatenate(predictions).flatten()

    def test(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        test_labels: np.ndarray,
        test_data: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Test the model.

        Args:
            model: PyTorch model
            test_loader: Test data loader
            test_labels: Ground truth labels
            test_data: Aggregate power data (optional)

        Returns:
            Dictionary with metrics
        """
        predictions = self.predict(model, test_loader)
        ground_truth = test_labels.copy()

        # Align based on model
        if self.model_name == 'cnn':
            offset = int(self.input_window_length / 2) - 1
            ground_truth = ground_truth[offset:]
            ground_truth = ground_truth[:len(predictions)]

        elif self.model_name == 'gru':
            offset = self.input_window_length - 1
            ground_truth = ground_truth[offset:]
            ground_truth = ground_truth[:len(predictions)]

        elif self.model_name == 'tcn':
            ground_truth = ground_truth[:len(predictions)]

        # Denormalize
        ground_truth = ground_truth * self.cutoff
        predictions = predictions * self.cutoff

        # Apply threshold and clip
        predictions[predictions < self.threshold] = 0
        predictions[predictions > self.cutoff] = self.cutoff

        # Calculate metrics
        mae = mean_absolute_error(ground_truth, predictions)

        gt_step = compute_step_function(ground_truth, self.threshold)
        pred_step = compute_step_function(predictions, self.threshold)

        f1, acc, precision, recall = acc_precision_recall_f1_score(gt_step, pred_step)

        print(f'MAE: {mae:.4f}')
        print(f'F1: {f1:.4f} | ACC: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}')

        return {
            'mae': mae,
            'f1': f1,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'predictions': predictions,
            'ground_truth': ground_truth
        }


def load_model(model: nn.Module, checkpoint_path: str, device: str = None) -> nn.Module:
    """
    Load model weights from checkpoint.

    Args:
        model: PyTorch model architecture
        checkpoint_path: Path to checkpoint file
        device: Device to load to

    Returns:
        Model with loaded weights
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    return model
