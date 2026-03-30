"""
src_pytorch/evaluator.py

Inference, metrics, and evaluation utilities for NILM models.

Sections
--------
1. Inference   — batched forward pass over a DataLoader
2. Metrics     — compute_status (duration-filtered ON/OFF), compute_metrics
3. Evaluation  — end-to-end test-set evaluation for CNN (Seq2Point) and TCN (Seq2Seq)
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, confusion_matrix


# =============================================================================
# 1. Inference
# =============================================================================

@torch.no_grad()
def run_predictions(model: nn.Module, data_loader, device: torch.device) -> np.ndarray:
    """Run batched inference and return a flat NumPy array of predictions.

    Parameters
    ----------
    model       : nn.Module
    data_loader : DataLoader
    device      : torch.device

    Returns
    -------
    np.ndarray — shape (N,), normalised predictions (not yet denormalised)
    """
    model.eval()
    preds = []
    for batch_x, _ in tqdm(data_loader, desc="Inference"):
        batch_x = batch_x.to(device)
        out = model(batch_x)
        preds.append(out.cpu().numpy())
    return np.concatenate(preds).flatten()


# =============================================================================
# 2. Metrics
# =============================================================================

def compute_status(
    power: np.ndarray,
    threshold: float,
    min_on: int,
    min_off: int,
    max_length: int = None,
) -> np.ndarray:
    """Compute binary ON/OFF status with duration-based denoising.

    Applies a power threshold to get an initial binary signal, then runs
    three passes:

    1. Removes short ON activations (< *min_on* samples).
    2. Fills short OFF gaps (<= *min_off* samples) to merge activations that
       belong to the same appliance cycle.
    3. Removes short ON activations again (< *min_on* samples) to discard
       any events that became too short after gap-filling (e.g. small ON
       segments created at the boundaries by Pass 2).

    Optionally a 4th pass removes any surviving ON event whose length is
    strictly less than *max_length*, providing a stricter final length check
    on top of the *min_on* filter.

    Parameters
    ----------
    power      : np.ndarray — power values in Watts (1-D)
    threshold  : float      — ON/OFF boundary in Watts
    min_on     : int        — minimum consecutive ON samples to keep as ON
    min_off    : int        — minimum consecutive OFF samples to keep as OFF
    max_length : int        — optional stricter minimum ON duration applied
                              after all other passes; events shorter than this
                              are removed (must be >= min_on to have any effect)

    Returns
    -------
    np.ndarray — binary status array (dtype int), same length as *power*
    """
    status = (power >= threshold).astype(int)

    # Pass 1 — remove ON activations shorter than min_on.
    i = 0
    while i < len(status):
        if status[i] == 1:
            j = i
            while j < len(status) and status[j] == 1:
                j += 1
            if j - i < min_on:
                status[i:j] = 0
            i = j
        else:
            i += 1

    # Pass 2 — fill short OFF gaps to merge nearby ON events.
    # A gap of exactly min_off samples is also filled (condition: <= min_off).
    i = 0
    while i < len(status):
        if status[i] == 0:
            j = i
            while j < len(status) and status[j] == 0:
                j += 1
            if j - i <= min_off:
                status[i:j] = 1
            i = j
        else:
            i += 1

    # Pass 3 — remove any ON activations still shorter than min_on.
    # Catches boundary artefacts or undersized events created by Pass 2.
    i = 0
    while i < len(status):
        if status[i] == 1:
            j = i
            while j < len(status) and status[j] == 1:
                j += 1
            if j - i < min_on:
                status[i:j] = 0
            i = j
        else:
            i += 1

    # Pass 4 (optional) — stricter final length filter.
    if max_length is not None:
        i = 0
        while i < len(status):
            if status[i] == 1:
                j = i
                while j < len(status) and status[j] == 1:
                    j += 1
                if j - i < max_length:
                    status[i:j] = 0
                i = j
            else:
                i += 1

    return status


def compute_metrics(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
    min_on: int = None,
    min_off: int = None,
    max_length: int = None,
) -> dict:
    """Compute standard NILM evaluation metrics.

    Parameters
    ----------
    ground_truth : np.ndarray — actual appliance power values in Watts
    predictions  : np.ndarray — predicted power values in Watts
    threshold    : float      — ON/OFF decision boundary in Watts
    min_on       : int        — if provided together with *min_off*, also
                               computes ``f1_complex`` using duration-filtered
                               status (see :func:`compute_status`)
    min_off      : int        — minimum OFF-duration for ``f1_complex``
    max_length   : int        — optional stricter final length filter passed
                               to :func:`compute_status`

    Returns
    -------
    dict with keys:
        mae, f1, accuracy, precision, recall,
        tp, tn, fp, fn,
        total_gt_energy_wh, total_pred_energy_wh, energy_error_percent
        and optionally f1_complex (when min_on and min_off are given)
    """
    mae = mean_absolute_error(ground_truth, predictions)

    gt_binary   = (ground_truth >= threshold).astype(int)
    pred_binary = (predictions  >= threshold).astype(int)

    tn, fp, fn, tp_val = confusion_matrix(gt_binary, pred_binary, labels=[0, 1]).ravel()

    accuracy  = (tp_val + tn) / (tp_val + tn + fp + fn)
    precision = tp_val / max(tp_val + fp, 1e-9)
    recall    = tp_val / max(tp_val + fn, 1e-9)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    gt_energy   = np.sum(ground_truth) / 3600   # Wh (10-s samples → /3600)
    pred_energy = np.sum(predictions)  / 3600
    energy_err  = abs(gt_energy - pred_energy) / max(gt_energy, 1e-9) * 100

    result = {
        'mae'                 : mae,
        'f1'                  : f1,
        'accuracy'            : accuracy,
        'precision'           : precision,
        'recall'              : recall,
        'tp'                  : int(tp_val),
        'tn'                  : int(tn),
        'fp'                  : int(fp),
        'fn'                  : int(fn),
        'total_gt_energy_wh'  : gt_energy,
        'total_pred_energy_wh': pred_energy,
        'energy_error_percent': energy_err,
    }

    if min_on is not None and min_off is not None:
        gt_status   = compute_status(ground_truth, threshold, min_on, min_off, max_length)
        pred_status = compute_status(predictions,  threshold, min_on, min_off, max_length)
        tn2, fp2, fn2, tp2 = confusion_matrix(gt_status, pred_status, labels=[0, 1]).ravel()
        prec2  = tp2 / max(tp2 + fp2, 1e-9)
        rec2   = tp2 / max(tp2 + fn2, 1e-9)
        result['f1_complex'] = 2 * prec2 * rec2 / max(prec2 + rec2, 1e-9)

    return result


# =============================================================================
# 3. Evaluation
# =============================================================================

def evaluate_model(
    model: nn.Module,
    data_loader,
    model_name: str,
    cutoff: float,
    threshold: float,
    device: torch.device,
    input_window_length: int = None,
    min_on: int = None,
    min_off: int = None,
    max_length: int = None,
) -> tuple:
    """Evaluate a NILM model on the test split.

    Handles ground-truth alignment automatically:

    * **CNN (Seq2Point)** — predictions are centre-point estimates so the
      ground-truth array is offset by ``int(input_window_length / 2) - 1``
      samples before comparison.
    * **TCN (Seq2Seq)** — predictions cover the full sequence, so the
      ground-truth array is simply front-truncated to match prediction length.

    Parameters
    ----------
    model               : nn.Module
    data_loader         : SimpleNILMDataLoader — must expose ``.test`` and
                          ``.test_labels``
    model_name          : str   — ``'cnn'`` or ``'tcn'``
    cutoff              : float — appliance power cutoff in Watts (for denormalisation)
    threshold           : float — ON/OFF boundary in Watts
    device              : torch.device
    input_window_length : int   — required for CNN alignment; ignored for TCN
    min_on              : int   — minimum ON-duration for status computation
    min_off             : int   — minimum OFF-duration for status computation
    max_length          : int   — optional stricter final length filter

    Returns
    -------
    (metrics, gt, pred, gt_status, pred_status)
        metrics    : dict — see :func:`compute_metrics`
        gt         : np.ndarray — denormalised ground truth in Watts
        pred       : np.ndarray — denormalised + clipped predictions in Watts
        gt_status  : np.ndarray or None — binary status for ground truth
        pred_status: np.ndarray or None — binary status for predictions
    """
    predictions_norm = run_predictions(model, data_loader.test, device)
    gt_norm = data_loader.test_labels.copy()

    if model_name == 'cnn':
        if input_window_length is None:
            raise ValueError("input_window_length is required for CNN evaluation.")
        offset = int(input_window_length / 2) - 1
        gt_norm = gt_norm[offset:][:len(predictions_norm)]
    elif model_name == 'gru':
        if input_window_length is None:
            raise ValueError("input_window_length is required for GRU evaluation.")
        offset = input_window_length - 1
        gt_norm = gt_norm[offset:][:len(predictions_norm)]
    else:  # tcn
        gt_norm = gt_norm[:len(predictions_norm)]

    gt   = gt_norm          * cutoff
    pred = predictions_norm * cutoff

    pred[pred < threshold] = 0
    pred[pred > cutoff]    = cutoff

    metrics = compute_metrics(gt, pred, threshold, min_on=min_on, min_off=min_off, max_length=max_length)

    gt_status   = None
    pred_status = None
    if min_on is not None and min_off is not None:
        gt_status   = compute_status(gt,   threshold, min_on, min_off, max_length)
        pred_status = compute_status(pred, threshold, min_on, min_off, max_length)

    return metrics, gt, pred, gt_status, pred_status
