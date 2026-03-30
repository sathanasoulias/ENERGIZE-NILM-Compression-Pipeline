"""
src_pytorch/pipeline.py

End-to-end pipeline steps for the ENERGIZE NILM optimisation workflow:

  1. run_training       — train from scratch with early stopping
  2. run_evaluation     — load checkpoint and evaluate on test split
  3. run_pruning        — structured channel pruning + test evaluation
  4. run_finetuning     — recovery training after pruning + test evaluation
  5. run_quantization   — TFLite INT8 conversion + test evaluation (TCN only)
  6. save_pipeline_results — write comparative CSV

All heavy lifting is delegated to the specialist modules:
  trainer.py   — training loop and callbacks
  pruner.py    — model statistics and structured pruning
  evaluator.py — inference, metrics, and evaluation
  quantizer.py — TFLite INT8 conversion (lazy import, TF is optional)
"""

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .models import CNN_NILM, GRU_NILM, TCN_NILM
from .data_loader import SimpleNILMDataLoader
from .trainer import Trainer
from .config import get_model_config, get_appliance_params, TRAINING, CALLBACKS
from .utils import count_parameters
from .pruner import (
    apply_torch_pruning,
    apply_unstructured_pruning,
    remove_pruning_masks,
    get_model_sparsity,
    get_model_stats,
    param_ratio_to_channel_ratio,
)
from .evaluator import (
    compute_status,
    evaluate_model,
)


# =============================================================================
# 1. Model factory
# =============================================================================

def build_nilm_model(model_name: str, model_config: dict) -> nn.Module:
    """Instantiate a NILM model from its configuration dict.

    Parameters
    ----------
    model_name   : ``'cnn'``, ``'gru'``, or ``'tcn'``
    model_config : dict returned by :func:`~src_pytorch.config.get_model_config`

    Returns
    -------
    nn.Module — randomly initialised model
    """
    window = model_config['input_window_length']

    if model_name == 'cnn':
        return CNN_NILM(input_window_length=window)
    if model_name == 'gru':
        return GRU_NILM(input_window_length=window)
    if model_name == 'tcn':
        return TCN_NILM(
            input_window_length=window,
            depth=model_config.get('depth', 9),
            nb_filters=model_config.get('nb_filters'),
            dropout=model_config.get('dropout', 0.2),
            stacks=model_config.get('stacks', 1),
        )
    raise ValueError(f"Unknown model: '{model_name}'. Choose from cnn, gru, tcn.")


# =============================================================================
# 2. Data loader factory
# =============================================================================

def get_data_loader(
    data_dir,
    model_name: str,
    model_config: dict,
    train: bool = True,
) -> SimpleNILMDataLoader:
    """Build and return a :class:`~src_pytorch.data_loader.SimpleNILMDataLoader`.

    Parameters
    ----------
    data_dir     : str or Path — directory containing ``training_.csv`` etc.
    model_name   : ``'cnn'``, ``'gru'``, or ``'tcn'``
    model_config : dict from :func:`~src_pytorch.config.get_model_config`
    train        : if ``True``, builds train + val + test loaders;
                   if ``False``, test loader only (saves memory during eval)
    """
    return SimpleNILMDataLoader(
        data_dir=str(data_dir),
        model_name=model_name,
        batch_size=model_config['batch_size'],
        input_window_length=model_config['input_window_length'],
        train=train,
        num_workers=0,
    )


# =============================================================================
# 3. Training
# =============================================================================

def run_training(
    dataset: str,
    appliance: str,
    model_name: str,
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> Path:
    """Train a NILM model from scratch.

    Uses the global :data:`~src_pytorch.config.TRAINING` and
    :data:`~src_pytorch.config.CALLBACKS` hyperparameters.

    Parameters
    ----------
    dataset    : ``'plegma'`` or ``'refit'``
    appliance  : e.g. ``'boiler'``
    model_name : ``'cnn'``, ``'gru'``, or ``'tcn'``
    data_dir   : path to processed CSV files
    output_dir : root output directory for this experiment
    device     : PyTorch device

    Returns
    -------
    Path — path to the saved best-model checkpoint (``.pt``)
    """
    model_config = get_model_config(model_name)
    data_loader  = get_data_loader(data_dir, model_name, model_config, train=True)

    print(f"\n{'='*60}")
    print(f"  Training  |  {model_name.upper()}  |  {appliance}  |  {dataset.upper()}")
    print(f"{'='*60}")
    print(f"  Train batches : {len(data_loader.train)}")
    print(f"  Val   batches : {len(data_loader.val)}")

    model = build_nilm_model(model_name, model_config).to(device)
    print(f"  Parameters    : {count_parameters(model):,}\n")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAINING['learning_rate'],
        betas=(TRAINING['beta_1'], TRAINING['beta_2']),
        eps=TRAINING['epsilon'],
    )

    checkpoint_dir  = output_dir / 'checkpoint'
    tensorboard_dir = output_dir / 'tensorboard'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        device=str(device),
    )
    trainer.setup_callbacks(
        checkpoint_dir=str(checkpoint_dir),
        tensorboard_dir=str(tensorboard_dir),
        early_stopping_patience=CALLBACKS['early_stopping']['patience'],
        early_stopping_min_delta=CALLBACKS['early_stopping']['min_delta'],
    )

    print(f"  Starting training (max {TRAINING['epochs']} epochs) ...")
    print(f"  Checkpoint dir : {checkpoint_dir}")
    print(f"  TensorBoard dir: {tensorboard_dir}\n")

    trainer.fit(
        train_loader=data_loader.train,
        val_loader=data_loader.val,
        epochs=TRAINING['epochs'],
        verbose=True,
    )

    return checkpoint_dir / 'model.pt'


# =============================================================================
# 4. Evaluation
# =============================================================================

def run_evaluation(
    dataset: str,
    appliance: str,
    model_name: str,
    checkpoint_path: Path,
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
    label: str = 'Baseline',
) -> dict:
    """Load a checkpoint and evaluate on the held-out test split.

    Parameters
    ----------
    dataset         : ``'plegma'`` or ``'refit'``
    appliance       : e.g. ``'boiler'``
    model_name      : ``'cnn'``, ``'gru'``, or ``'tcn'``
    checkpoint_path : path to ``.pt`` file
    data_dir        : path to processed CSV files
    output_dir      : root output directory (unused here, kept for API symmetry)
    device          : PyTorch device
    label           : display label used in printed output

    Returns
    -------
    dict — metrics: mae, f1, accuracy, precision, recall, energy_error_percent
    """
    model_config     = get_model_config(model_name)
    appliance_params = get_appliance_params(dataset, appliance)
    data_loader      = get_data_loader(data_dir, model_name, model_config, train=False)

    print(f"\n{'='*60}")
    print(f"  Evaluation [{label}]  |  {model_name.upper()}  |  {appliance}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    model = build_nilm_model(model_name, model_config).to(device)
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
    model.eval()

    metrics, gt, pred, gt_status, pred_status = evaluate_model(
        model=model,
        data_loader=data_loader,
        model_name=model_name,
        cutoff=appliance_params['cutoff'],
        threshold=appliance_params['threshold'],
        device=device,
        input_window_length=model_config['input_window_length'],
        min_on=appliance_params.get('min_on'),
        min_off=appliance_params.get('min_off'),
        max_length=appliance_params.get('max_length'),
    )

    _print_metrics(metrics, label=label)
    _save_predictions_csv(output_dir, 'baseline', gt, pred, gt_status, pred_status)
    _save_metrics_csv(output_dir, dataset, appliance, model_name, label, metrics)
    return metrics


# =============================================================================
# 5. Pruning
# =============================================================================

def run_pruning(
    model: nn.Module,
    data_loader: SimpleNILMDataLoader,
    model_name: str,
    model_config: dict,
    appliance_params: dict,
    pruning_ratio: float,
    output_dir: Path,
    label: str,
    device: torch.device,
) -> tuple:
    """Apply global structured channel pruning and evaluate on the test split.

    Uses magnitude-based importance via ``torch_pruning.MetaPruner``.
    The output Linear layer (CNN/GRU) is protected from pruning automatically.

    Parameters
    ----------
    model           : fresh model loaded from baseline checkpoint
    data_loader     : loader with ``.train`` and ``.test`` splits
    model_name      : ``'cnn'``, ``'gru'``, or ``'tcn'``
    model_config    : dict from :func:`~src_pytorch.config.get_model_config`
    appliance_params: dict from :func:`~src_pytorch.config.get_appliance_params`
    pruning_ratio   : target *parameter* reduction fraction (e.g. ``0.75``)
    output_dir      : root output directory; pruned checkpoint saved under
                      ``<output_dir>/models/``
    label           : experiment label used in filename and printed output
    device          : PyTorch device

    Returns
    -------
    (pruned_model, metrics, checkpoint_path)
    """
    pct        = int(pruning_ratio * 100)
    models_dir = output_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    window      = model_config['input_window_length']
    dummy_input = torch.randn(1, window).to(device)

    # Protect the output Linear from being pruned.
    # CNN/GRU: last Linear has out_features=1 (Seq2Point).
    # TCN: no Linear layers, so window_size value is irrelevant.
    args = SimpleNamespace(window_size=1 if model_name in ('cnn', 'gru') else window)

    print(f"\n{'='*60}")
    print(f"  Pruning [{label}]  |  target param reduction: {pct}%")
    print(f"{'='*60}")

    # Pre-convert param ratio → channel ratio, matching the notebook convention:
    #   _channel_ratio = param_ratio_to_channel_ratio(PRUNING_RATIO)
    # apply_torch_pruning then converts it once more internally, replicating
    # the exact same double-conversion that 03_pruning.ipynb performs.
    channel_ratio = pruning_ratio
    model, _ = apply_torch_pruning(model, args, dummy_input, channel_ratio)

    params, macs, mb = get_model_stats(model, dummy_input)
    print(f"\n  Post-prune stats — Params: {params:,}  MACs: {macs:,}  Size: {mb:.3f} MB")

    metrics, gt, pred, gt_status, pred_status = evaluate_model(
        model=model,
        data_loader=data_loader,
        model_name=model_name,
        cutoff=appliance_params['cutoff'],
        threshold=appliance_params['threshold'],
        device=device,
        input_window_length=window,
        min_on=appliance_params.get('min_on'),
        min_off=appliance_params.get('min_off'),
        max_length=appliance_params.get('max_length'),
    )
    _print_metrics(metrics, label=f'Pruned {pct}%')
    _save_predictions_csv(output_dir, f'pruned_{pct}pct', gt, pred, gt_status, pred_status)

    ckpt_path = models_dir / f'{label}_pruned_{pct}pct.pt'
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Pruned checkpoint saved: {ckpt_path.name}")

    return model, metrics, ckpt_path


# =============================================================================
# 5b. Unstructured Pruning
# =============================================================================

def run_unstructured_pruning(
    model: nn.Module,
    data_loader: SimpleNILMDataLoader,
    model_name: str,
    model_config: dict,
    appliance_params: dict,
    sparsity: float,
    output_dir: Path,
    label: str,
    device: torch.device,
) -> tuple:
    """Apply global L1 unstructured pruning and evaluate on the test split.

    Unlike structured pruning, unstructured pruning:
    - Works with **all** model types including GRU.
    - Does **not** change the model architecture or tensor shapes.
    - Zeros the weights with the smallest absolute values globally across all
      Conv1d, Linear, and GRU weight tensors.
    - Attaches forward-pre-hooks so the zeroed weights remain zero during
      subsequent fine-tuning (call :func:`remove_pruning_masks` to make
      permanent before saving).

    Parameters
    ----------
    model           : fresh model loaded from baseline checkpoint
    data_loader     : loader with ``.train`` and ``.test`` splits
    model_name      : ``'cnn'``, ``'gru'``, or ``'tcn'``
    model_config    : dict from :func:`~src_pytorch.config.get_model_config`
    appliance_params: dict from :func:`~src_pytorch.config.get_appliance_params`
    sparsity        : fraction of weights to zero, e.g. ``0.75`` zeros 75 %
    output_dir      : root output directory; pruned checkpoint saved under
                      ``<output_dir>/models/``
    label           : experiment label used in filename and printed output
    device          : PyTorch device

    Returns
    -------
    (pruned_model, metrics, checkpoint_path)
        pruned_model    — model with pruning hooks still active (ready for
                          fine-tuning; call remove_pruning_masks before saving)
        metrics         — evaluation metrics dict
        checkpoint_path — ``.pt`` file with masks removed and weights baked in
    """
    pct        = int(sparsity * 100)
    models_dir = output_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Unstructured Pruning [{label}]  |  target sparsity: {pct}%")
    print(f"{'='*60}")

    model = apply_unstructured_pruning(model, amount=sparsity)

    # Model shape is unchanged — report sparsity instead of MACs delta
    params, macs, mb = get_model_stats(
        model,
        torch.randn(1, model_config['input_window_length']).to(device),
    )
    actual_sparsity = get_model_sparsity(model)
    print(
        f"\n  Post-prune stats — Params: {params:,}  MACs: {macs:,}  "
        f"Size: {mb:.3f} MB  Sparsity: {actual_sparsity * 100:.1f}%"
    )

    metrics, gt, pred, gt_status, pred_status = evaluate_model(
        model=model,
        data_loader=data_loader,
        model_name=model_name,
        cutoff=appliance_params['cutoff'],
        threshold=appliance_params['threshold'],
        device=device,
        input_window_length=model_config['input_window_length'],
        min_on=appliance_params.get('min_on'),
        min_off=appliance_params.get('min_off'),
        max_length=appliance_params.get('max_length'),
    )
    _print_metrics(metrics, label=f'Unstructured Pruned {pct}%')
    _save_predictions_csv(
        output_dir, f'unstructured_pruned_{pct}pct', gt, pred, gt_status, pred_status
    )

    # Save a clean checkpoint (masks removed, zeros baked in)
    remove_pruning_masks(model)
    ckpt_path = models_dir / f'{label}_unstructured_pruned_{pct}pct.pt'
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Pruned checkpoint saved: {ckpt_path.name}")

    return model, metrics, ckpt_path


# =============================================================================
# 6. Fine-tuning
# =============================================================================

def run_finetuning(
    model: nn.Module,
    data_loader: SimpleNILMDataLoader,
    model_name: str,
    model_config: dict,
    appliance_params: dict,
    pruning_ratio: float,
    epochs: int,
    lr: float,
    output_dir: Path,
    label: str,
    device: torch.device,
) -> tuple:
    """Fine-tune a pruned model on the training split.

    Uses MSE loss and Adam. No early stopping — runs exactly *epochs* epochs.

    Parameters
    ----------
    model           : pruned model returned by :func:`run_pruning`
    data_loader     : same loader used for pruning (has ``.train`` and ``.test``)
    model_name      : ``'cnn'``, ``'gru'``, or ``'tcn'``
    model_config    : dict from :func:`~src_pytorch.config.get_model_config`
    appliance_params: dict from :func:`~src_pytorch.config.get_appliance_params`
    pruning_ratio   : same fraction used in :func:`run_pruning` — embedded in
                      the checkpoint filename, e.g. ``pruned_50pct_finetuned.pt``
    epochs          : number of fine-tuning epochs
    lr              : Adam learning rate
    output_dir      : root output directory; checkpoint saved under
                      ``<output_dir>/models/``
    label           : experiment label used in filename and printed output
    device          : PyTorch device

    Returns
    -------
    (model, metrics, checkpoint_path)
    """
    pct        = int(pruning_ratio * 100)
    models_dir = output_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    print(f"\n{'='*60}")
    print(f"  Fine-tuning [{label}]  |  {epochs} epoch(s)  |  LR={lr}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch_x, batch_y in tqdm(data_loader.train, desc=f"  Epoch {epoch}/{epochs}"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)

            # Align target shape to model output
            # CNN/GRU : output (B, 1),          target (B,)         → (B, 1)
            # TCN     : output (B, seq_len, 1), target (B, seq_len) → (B, seq_len, 1)
            if model_name in ('cnn', 'gru') and batch_y.dim() == 1:
                batch_y = batch_y.unsqueeze(1)
            elif model_name == 'tcn' and batch_y.dim() == 2:
                batch_y = batch_y.unsqueeze(-1)

            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        print(f"  Epoch {epoch}/{epochs} — avg MSE: {total_loss / n_batches:.6f}")

    model.eval()
    metrics, gt, pred, gt_status, pred_status = evaluate_model(
        model=model,
        data_loader=data_loader,
        model_name=model_name,
        cutoff=appliance_params['cutoff'],
        threshold=appliance_params['threshold'],
        device=device,
        input_window_length=model_config['input_window_length'],
        min_on=appliance_params.get('min_on'),
        min_off=appliance_params.get('min_off'),
        max_length=appliance_params.get('max_length'),
    )
    _print_metrics(metrics, label=f'Fine-tuned {epochs}ep')
    _save_predictions_csv(
        output_dir, f'pruned_{pct}pct_finetuned', gt, pred, gt_status, pred_status
    )

    # Naming matches notebook 03_pruning.ipynb: {label}_pruned_{pct}pct_finetuned.pt
    ckpt_path = models_dir / f'{label}_pruned_{pct}pct_finetuned.pt'
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Fine-tuned checkpoint saved: {ckpt_path.name}")

    return model, metrics, ckpt_path


# =============================================================================
# 7. Quantization
# =============================================================================

def run_quantization(
    model_name: str,
    model_config: dict,
    appliance_params: dict,
    baseline_ckpt: Path,
    pruning_ratio: float,
    finetuned_ckpt: Path,
    data_loader: SimpleNILMDataLoader,
    output_dir: Path,
    label: str,
    n_calib_batches: int = 100,
) -> dict | None:
    """Convert the fine-tuned pruned model to TFLite full-integer INT8.

    Currently supports **TCN only**.  For CNN or GRU a message is printed and
    ``None`` is returned — quantization for those architectures is planned for
    a later version.

    TensorFlow is imported lazily; if it is not installed the function prints
    install instructions and returns ``None``.

    Parameters
    ----------
    model_name       : ``'cnn'``, ``'gru'``, or ``'tcn'``
    model_config     : dict from :func:`~src_pytorch.config.get_model_config`
    appliance_params : dict from :func:`~src_pytorch.config.get_appliance_params`
    baseline_ckpt    : path to the original (pre-pruning) checkpoint
    pruning_ratio    : same parameter-reduction ratio used in :func:`run_pruning`
    finetuned_ckpt   : path to the fine-tuned checkpoint from :func:`run_finetuning`
    data_loader      : loader with ``.train`` (calibration) and ``.test`` (eval)
    output_dir       : root output directory; ``.tflite`` saved under
                       ``<output_dir>/models/``
    label            : experiment label used in the ``.tflite`` filename,
                       e.g. ``tcn_boiler``
    n_calib_batches  : number of training batches used for INT8 calibration

    Returns
    -------
    dict | None — metrics dict (mae, f1, …) or ``None`` if skipped
    """
    if model_name != 'tcn':
        print(
            f"\n  [Quantization] {model_name.upper()} quantization is not yet "
            "supported — to be implemented in a later version."
        )
        return None

    try:
        from src_pytorch.quantizer import (
            rebuild_pruned_tcn,
            read_pruned_channels,
            build_tcn_keras,
            transfer_weights,
            validate_weight_transfer,
            convert_to_tflite_int8,
            evaluate_tflite,
        )
    except ImportError:
        print(
            "\n  [Quantization] TensorFlow not found. Install with:\n"
            "    pip install tensorflow\n"
            "  Skipping quantization."
        )
        return None

    pct = int(pruning_ratio * 100)

    print(f"\n{'='*60}")
    print(f"  Quantization  |  TCN  |  TFLite INT8")
    print(f"{'='*60}")

    cpu = torch.device('cpu')

    # Map src_pytorch config keys → the cfg dict expected by quantizer helpers
    tcn_cfg = {
        'window'          : model_config['input_window_length'],
        'depth'           : model_config.get('depth', 9),
        'filters'         : model_config.get('nb_filters'),
        'dropout'         : model_config.get('dropout', 0.2),
        'stacks'          : model_config.get('stacks', 1),
        'args_window_size': model_config['input_window_length'],
    }

    # Pre-convert param ratio → channel ratio, matching notebook 04_quantization cell-9:
    #   pruning_ratio = _channel_ratio   # internal channel ratio, not PRUNING_RATIO
    # rebuild_pruned_tcn calls apply_torch_pruning, which converts once more internally —
    # the double-conversion replicates the exact behaviour of the notebooks.
    channel_ratio = pruning_ratio

    # Reconstruct pruned architecture and load fine-tuned weights
    pt_model = rebuild_pruned_tcn(
        cfg=tcn_cfg,
        baseline_ckpt=baseline_ckpt,
        pruning_ratio=channel_ratio,
        finetuned_ckpt_path=finetuned_ckpt,
        device=cpu,
    )

    # Build matching Keras model and transfer weights
    initial_ch, block_filters = read_pruned_channels(
        pt_model, depth=tcn_cfg['depth'], stacks=tcn_cfg['stacks']
    )
    tf_model = build_tcn_keras(
        initial_ch=initial_ch,
        block_filters=block_filters,
        depth=tcn_cfg['depth'],
        stacks=tcn_cfg['stacks'],
        dropout_rate=tcn_cfg['dropout'],
        seq_len=tcn_cfg['window'],
    )

    transfer_weights(
        tf_model, pt_model.state_dict(), tcn_cfg['depth'], tcn_cfg['stacks']
    )

    print("  Validating weight transfer...")
    validate_weight_transfer(pt_model, tf_model, data_loader, cpu)

    # Convert to TFLite INT8
    models_dir  = output_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    # Naming matches notebook 04: {label}_pruned_{pct}pct_int8.tflite
    tflite_path = models_dir / f'{label}_pruned_{pct}pct_int8.tflite'

    tflite_path = convert_to_tflite_int8(
        tf_model=tf_model,
        data_loader=data_loader,
        window=tcn_cfg['window'],
        n_calib_batches=n_calib_batches,
        out_path=tflite_path,
    )

    # Evaluate on test split
    print("  Evaluating TFLite INT8 on test set...")
    metrics, gt, pred, gt_status, pred_status = evaluate_tflite(
        tflite_path=tflite_path,
        data_loader=data_loader,
        window=tcn_cfg['window'],
        cutoff=appliance_params['cutoff'],
        threshold=appliance_params['threshold'],
        min_on=appliance_params.get('min_on'),
        min_off=appliance_params.get('min_off'),
        max_length=appliance_params.get('max_length'),
    )
    _print_metrics(metrics, label='TFLite INT8')
    pct = int(pruning_ratio * 100)
    _save_predictions_csv(output_dir, f'pruned_{pct}pct_int8', gt, pred, gt_status, pred_status)

    return metrics


# =============================================================================
# 8. Results I/O
# =============================================================================

def save_pipeline_results(
    rows: list,
    output_dir: Path,
    appliance: str,
    model_name: str,
) -> None:
    """Write per-stage metrics to a comparative CSV file.

    Parameters
    ----------
    rows       : list of dicts, each containing a ``'label'`` key plus the
                 standard metric keys returned by :func:`~src_pytorch.pruner.compute_metrics`
    output_dir : directory to write the CSV into
    appliance  : used in the output filename
    model_name : used in the output filename
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f'{appliance}_{model_name}_pipeline_results.csv'

    fieldnames = [
        'Stage', 'Params', 'MACs', 'MB',
        'MAE', 'F1', 'F1_Complex', 'Precision', 'Recall', 'Accuracy', 'Energy_Error_%',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            f1c = row.get('f1_complex')
            writer.writerow({
                'Stage'          : row['label'],
                'Params'         : row.get('params', ''),
                'MACs'           : row.get('macs',   ''),
                'MB'             : row.get('mb',     ''),
                'MAE'            : round(row['mae'],                  4),
                'F1'             : round(row['f1'],                   4),
                'F1_Complex'     : round(f1c, 4) if f1c is not None else '',
                'Precision'      : round(row['precision'],            4),
                'Recall'         : round(row['recall'],               4),
                'Accuracy'       : round(row['accuracy'],             4),
                'Energy_Error_%' : round(row['energy_error_percent'], 2),
            })

    print(f"\n  Pipeline results saved to: {csv_path}")


# =============================================================================
# Internal helpers
# =============================================================================

def _save_predictions_csv(
    output_dir: Path,
    stage_slug: str,
    gt: np.ndarray,
    pred: np.ndarray,
    gt_status: np.ndarray = None,
    pred_status: np.ndarray = None,
) -> None:
    """Save per-sample predictions to a CSV file under ``<output_dir>/predictions/``.

    Parameters
    ----------
    output_dir  : experiment root directory
    stage_slug  : short identifier used in the filename,
                  e.g. ``'baseline'``, ``'pruned_50pct'``, ``'tflite_int8'``
    gt          : denormalised ground truth in Watts
    pred        : denormalised predictions in Watts
    gt_status   : binary ON/OFF status for ground truth (optional)
    pred_status : binary ON/OFF status for predictions (optional)
    """
    pred_dir = output_dir / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    csv_path = pred_dir / f'{stage_slug}_predictions.csv'

    has_status = gt_status is not None and pred_status is not None

    cols   = [gt, pred]
    header = ['ground_truth_W', 'prediction_W']
    if has_status:
        cols  += [gt_status.astype(np.int8), pred_status.astype(np.int8)]
        header += ['ground_truth_status', 'predicted_status']

    np.savetxt(
        csv_path,
        np.column_stack(cols),
        delimiter=',',
        header=','.join(header),
        comments='',
        fmt=['%.4f', '%.4f'] + (['%d', '%d'] if has_status else []),
    )
    print(f"  Predictions saved: {csv_path.name}")


def _save_metrics_csv(
    output_dir: Path,
    dataset: str,
    appliance: str,
    model_name: str,
    label: str,
    metrics: dict,
) -> None:
    """Save a single-row metrics CSV to ``<output_dir>/metrics/``.

    Mirrors the format produced by notebook 02 (``{appliance}_results.csv``),
    extended with a ``Stage`` column so multiple calls don't collide.

    Parameters
    ----------
    output_dir : experiment root directory
    dataset    : dataset name (e.g. ``'plegma'``)
    appliance  : appliance name (e.g. ``'boiler'``)
    model_name : model architecture (e.g. ``'tcn'``)
    label      : stage label used in the filename and as the ``Stage`` column
    metrics    : dict returned by :func:`~src_pytorch.evaluator.compute_metrics`
    """
    import pandas as pd

    metrics_dir = output_dir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)

    slug = label.lower().replace(' ', '_').replace('%', 'pct').replace('+', '').replace('__', '_')
    csv_path = metrics_dir / f'{appliance}_{slug}_results.csv'

    f1c = metrics.get('f1_complex')
    row = {
        'Stage'               : label,
        'Model'               : model_name,
        'Appliance'           : appliance,
        'Dataset'             : dataset,
        'MAE'                 : round(metrics['mae'],                  4),
        'F1'                  : round(metrics['f1'],                   4),
        'F1_Complex'          : round(f1c, 4) if f1c is not None else '',
        'Accuracy'            : round(metrics['accuracy'],             4),
        'Precision'           : round(metrics['precision'],            4),
        'Recall'              : round(metrics['recall'],               4),
        'GT_Energy_Wh'        : round(metrics['total_gt_energy_wh'],   2),
        'Pred_Energy_Wh'      : round(metrics['total_pred_energy_wh'], 2),
        'Energy_Error_Percent': round(metrics['energy_error_percent'],  2),
    }
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    print(f"  Stage metrics saved: {csv_path.name}")


def _print_metrics(
    metrics: dict,
    *,
    appliance: str = '',
    model_name: str = '',
    label: str = '',
) -> None:
    """Pretty-print a metrics dict."""
    parts  = [p for p in (appliance.upper(), model_name.upper(), label) if p]
    header = '  '.join(parts)
    print(f"\n  {'─'*40}")
    if header:
        print(f"  {header}")
    print(f"  {'─'*40}")
    print(f"  MAE            : {metrics['mae']:.4f} W")
    print(f"  F1             : {metrics['f1']:.4f}")
    if 'f1_complex' in metrics:
        print(f"  F1 Complex     : {metrics['f1_complex']:.4f}")
    print(f"  Accuracy       : {metrics['accuracy']:.4f}")
    print(f"  Precision      : {metrics['precision']:.4f}")
    print(f"  Recall         : {metrics['recall']:.4f}")
    print(f"  Energy Error % : {metrics['energy_error_percent']:.2f}")
    print(f"  {'─'*40}\n")
