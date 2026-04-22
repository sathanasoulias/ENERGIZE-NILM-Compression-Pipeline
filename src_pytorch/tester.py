# Copyright (c) 2026 Sotirios Athanasoulias. MIT License — see LICENSE for details.
"""
src_pytorch/tester.py

Lightweight inference helpers for notebook-based evaluation.

Classes
-------
SimpleTester — predict + align + denormalise + compute complex metrics
               (no Hydra cfg; pass parameters directly)

Functions
---------
load_model   — load model weights from a state-dict checkpoint
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .evaluator import compute_metrics, compute_status


class SimpleTester:
    """Tester for evaluating NILM models without a config framework.

    Accepts direct parameters so it can be used from notebooks and
    standalone scripts. Metric reporting uses the duration-filtered
    complex F1 / Precision / Recall (``f1_complex``, ``precision_complex``,
    ``recall_complex``) consistent with the rest of the pipeline.
    """

    def __init__(
        self,
        model_name: str,
        input_window_length: int,
        threshold: float,
        cutoff: float,
        min_on: int = None,
        min_off: int = None,
        min_committed_duration: int = None,
    ):
        """
        Parameters
        ----------
        model_name              : ``'cnn'``, ``'cnn_seq2seq'``, or ``'wavenet_tcn'``
        input_window_length     : input sequence length
        threshold               : ON/OFF boundary in Watts
        cutoff                  : normalisation ceiling in Watts
        min_on                  : minimum ON-duration (samples) for complex metrics
        min_off                 : minimum OFF-gap (samples) for complex metrics
        min_committed_duration  : optional stricter ON-duration filter
        """
        self.model_name             = model_name.lower()
        self.input_window_length    = input_window_length
        self.threshold              = threshold
        self.cutoff                 = cutoff
        self.min_on                 = min_on
        self.min_off                = min_off
        self.min_committed_duration = min_committed_duration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @torch.no_grad()
    def predict(self, model: nn.Module, data_loader: DataLoader) -> np.ndarray:
        """Run batched inference and return flat normalised predictions."""
        model.eval()
        model.to(self.device)
        preds = []
        for batch_x, _ in tqdm(data_loader, desc='Predicting'):
            batch_x = batch_x.to(self.device)
            preds.append(model(batch_x).cpu().numpy())
        return np.concatenate(preds).flatten()

    def test(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        test_labels: np.ndarray,
    ) -> dict:
        """Run inference, align ground truth, denormalise, and compute metrics.

        Parameters
        ----------
        model       : PyTorch model
        test_loader : test DataLoader
        test_labels : normalised ground-truth array (full sequence length)

        Returns
        -------
        dict with keys: metrics (dict), predictions (np.ndarray),
        ground_truth (np.ndarray), gt_status, pred_status
        """
        predictions_norm = self.predict(model, test_loader)
        gt_norm = test_labels.copy()

        if self.model_name == 'cnn':
            offset = int(self.input_window_length / 2) - 1
            gt_norm = gt_norm[offset:][:len(predictions_norm)]
        else:  # cnn_seq2seq, wavenet_tcn — full sequence output
            gt_norm = gt_norm[:len(predictions_norm)]

        gt   = gt_norm          * self.cutoff
        pred = predictions_norm * self.cutoff

        pred[pred < self.threshold] = 0
        pred[pred > self.cutoff]    = self.cutoff

        metrics = compute_metrics(
            gt, pred, self.threshold,
            min_on=self.min_on,
            min_off=self.min_off,
            min_committed_duration=self.min_committed_duration,
        )

        gt_status   = None
        pred_status = None
        if self.min_on is not None and self.min_off is not None:
            gt_status   = compute_status(gt,   self.threshold, self.min_on, self.min_off,
                                         self.min_committed_duration)
            pred_status = compute_status(pred, self.threshold, self.min_on, self.min_off,
                                         self.min_committed_duration)

        print(f'MAE        : {metrics["mae"]:.4f} W')
        print(f'Accuracy   : {metrics["accuracy"]:.4f}')
        if 'f1_complex' in metrics:
            print(f'F1 Complex : {metrics["f1_complex"]:.4f}')
            print(f'Precision  : {metrics["precision_complex"]:.4f}')
            print(f'Recall     : {metrics["recall_complex"]:.4f}')

        return {
            'metrics'     : metrics,
            'predictions' : pred,
            'ground_truth': gt,
            'gt_status'   : gt_status,
            'pred_status' : pred_status,
        }


def load_model(model: nn.Module, checkpoint_path: str, device: str = None) -> nn.Module:
    """Load model weights from a state-dict checkpoint.

    Parameters
    ----------
    model           : model architecture (uninitialised weights)
    checkpoint_path : path to ``.pt`` file saved with ``torch.save(model.state_dict(), ...)``
    device          : device string; defaults to CUDA if available, else CPU

    Returns
    -------
    nn.Module — model in eval mode with loaded weights
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    print(f'Model loaded from {checkpoint_path}')
    return model
