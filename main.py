"""
ENERGIZE — Main Training and Evaluation Script
================================================
ENERGIZE Project | DNN-based NILM Model Training

Runs the full pipeline:
  1. Load processed data (CSV files produced by data/data.py)
  2. Build the requested DNN model
  3. Train with early stopping and model checkpointing
  4. Save training history to CSV
  5. Evaluate on the held-out test house
  6. Save metrics and per-sample predictions to CSV

Usage
-----
    # Default: PLEGMA dataset, Boiler appliance, TCN model
    python main.py

    # Custom experiment
    python main.py --dataset plegma --appliance boiler --model tcn

    # Evaluate only (skip training)
    python main.py --eval-only --checkpoint outputs/tcn_boiler/checkpoint/model.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Add project root to path when running as a script
sys.path.insert(0, str(Path(__file__).parent))

from src_pytorch import (
    CNN_NILM, GRU_NILM, TCN_NILM,
    SimpleNILMDataLoader,
    Trainer,
    set_seeds,
    get_device,
    count_parameters,
)
from src_pytorch.config import (
    get_model_config,
    get_appliance_params,
    TRAINING,
    CALLBACKS,
)
from src_pytorch.evaluator import evaluate_model, compute_status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_model(model_name: str, model_config: dict) -> nn.Module:
    """Instantiate a NILM model from its configuration dict."""
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


def save_results(
    metrics: dict,
    output_dir: Path,
    dataset: str,
    appliance: str,
    model_name: str,
    gt: np.ndarray,
    pred: np.ndarray,
    threshold: float,
    min_on: int = None,
    min_off: int = None,
    max_length: int = None,
) -> None:
    """Write evaluation metrics and per-sample predictions to CSV.

    Matches the output format produced by notebook 02_evaluation.ipynb.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Metrics CSV ──────────────────────────────────────────────────────────
    f1c = metrics.get('f1_complex')
    row = {
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
    metrics_path = output_dir / f'{appliance}_results.csv'
    pd.DataFrame([row]).to_csv(metrics_path, index=False)
    print(f"  Metrics saved    : {metrics_path}")

    # ── Predictions CSV ──────────────────────────────────────────────────────
    errors = pred - gt
    pred_data = {
        'ground_truth_W': gt,
        'prediction_W'  : pred,
        'error'         : errors,
        'abs_error'     : np.abs(errors),
    }
    if min_on is not None and min_off is not None:
        pred_data['ground_truth_status'] = compute_status(
            gt, threshold, min_on, min_off, max_length).astype(int)
        pred_data['predicted_status'] = compute_status(
            pred, threshold, min_on, min_off, max_length).astype(int)

    predictions_path = output_dir / f'{appliance}_predictions.csv'
    pd.DataFrame(pred_data).to_csv(predictions_path, index=False)
    print(f"  Predictions saved: {predictions_path}")


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def train(
    dataset: str,
    appliance: str,
    model_name: str,
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> Path:
    """Train a NILM model and return the path to the best checkpoint."""
    model_config = get_model_config(model_name)

    # Data
    print(f"\n{'='*60}")
    print(f"  Loading data from: {data_dir}")
    print(f"{'='*60}")

    data_loader = SimpleNILMDataLoader(
        data_dir=str(data_dir),
        model_name=model_name,
        batch_size=model_config['batch_size'],
        input_window_length=model_config['input_window_length'],
        train=True,
        num_workers=0,
    )

    print(f"  Training batches  : {len(data_loader.train)}")
    print(f"  Validation batches: {len(data_loader.val)}")

    # Model
    model = build_model(model_name, model_config).to(device)
    print(f"\n  Model             : {model_name.upper()}")
    print(f"  Parameters        : {count_parameters(model):,}")

    # Optimiser and trainer
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

    print(f"\n  Starting training  (max {TRAINING['epochs']} epochs) ...")
    print(f"  Checkpoint dir    : {checkpoint_dir}")
    print(f"  TensorBoard dir   : {tensorboard_dir}")
    print(f"{'='*60}\n")

    history = trainer.fit(
        train_loader=data_loader.train,
        val_loader=data_loader.val,
        epochs=TRAINING['epochs'],
        verbose=True,
    )

    # Save training history (mirrors notebook 01)
    history_df = pd.DataFrame({
        'epoch'     : history.epochs,
        'train_loss': history.train_loss,
        'val_loss'  : history.val_loss,
        'train_mae' : history.train_mae,
        'val_mae'   : history.val_mae,
    })
    history_path = output_dir / 'training_history.csv'
    history_df.to_csv(history_path, index=False)
    print(f"\n  Training history saved: {history_path}")

    return checkpoint_dir / 'model.pt'


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate(
    dataset: str,
    appliance: str,
    model_name: str,
    checkpoint_path: Path,
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> dict:
    """Load the best checkpoint and evaluate on the held-out test house."""
    model_config     = get_model_config(model_name)
    appliance_params = get_appliance_params(dataset, appliance)

    min_on     = appliance_params.get('min_on')
    min_off    = appliance_params.get('min_off')
    max_length = appliance_params.get('max_length')

    print(f"\n{'='*60}")
    print(f"  Evaluating checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    # Load test data (train=False loads only the test split)
    data_loader = SimpleNILMDataLoader(
        data_dir=str(data_dir),
        model_name=model_name,
        batch_size=model_config['batch_size'],
        input_window_length=model_config['input_window_length'],
        train=False,
        num_workers=0,
    )

    # Load model
    model = build_model(model_name, model_config).to(device)
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
    model.eval()
    print(f"  Model loaded: {checkpoint_path.name}")

    # Run inference and compute metrics (mirrors notebook 02)
    metrics, gt, pred, _, _ = evaluate_model(
        model=model,
        data_loader=data_loader,
        model_name=model_name,
        cutoff=appliance_params['cutoff'],
        threshold=appliance_params['threshold'],
        device=device,
        input_window_length=model_config['input_window_length'],
        min_on=min_on,
        min_off=min_off,
        max_length=max_length,
    )

    # Print results
    f1c = metrics.get('f1_complex')
    print(f"\n  {'─'*40}")
    print(f"  {appliance.upper()}  |  {model_name.upper()}  |  {dataset.upper()}")
    print(f"  {'─'*40}")
    print(f"  MAE            : {metrics['mae']:.4f} W")
    print(f"  F1-Complex     : {f1c:.4f}" if f1c is not None else "  F1-Complex     : —")
    print(f"  Accuracy       : {metrics['accuracy']:.4f}")
    print(f"  Energy Error % : {metrics['energy_error_percent']:.2f}")
    print(f"  {'─'*40}\n")

    # Save metrics and predictions (mirrors notebook 02)
    save_results(
        metrics=metrics,
        output_dir=output_dir / 'metrics',
        dataset=dataset,
        appliance=appliance,
        model_name=model_name,
        gt=gt,
        pred=pred,
        threshold=appliance_params['threshold'],
        min_on=min_on,
        min_off=min_off,
        max_length=max_length,
    )

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='ENERGIZE NILM — Train and evaluate a DNN disaggregation model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset',     '-d', type=str, default='plegma',
                        choices=['refit', 'plegma'], help='Dataset to use')
    parser.add_argument('--appliance',   '-a', type=str, default='boiler',
                        help='Appliance to disaggregate')
    parser.add_argument('--model',       '-m', type=str, default='tcn',
                        choices=['cnn', 'gru', 'tcn'], help='Model architecture')
    parser.add_argument('--data-root',         type=str, default='./data/processed',
                        help='Root directory containing processed CSV files')
    parser.add_argument('--output-root',       type=str, default='./outputs',
                        help='Root directory for checkpoints and results')
    parser.add_argument('--seed',              type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--eval-only',   action='store_true',
                        help='Skip training and run evaluation only')
    parser.add_argument('--checkpoint',        type=str, default=None,
                        help='Path to checkpoint for --eval-only mode')
    return parser.parse_args()


def main():
    args = parse_args()

    set_seeds(args.seed)
    device = get_device()

    data_dir   = Path(args.data_root)  / args.dataset  / args.appliance
    output_dir = Path(args.output_root) / f'{args.model}_{args.appliance}'

    print(f"\n{'='*60}")
    print(f"  ENERGIZE NILM")
    print(f"{'='*60}")
    print(f"  Dataset   : {args.dataset}")
    print(f"  Appliance : {args.appliance}")
    print(f"  Model     : {args.model.upper()}")
    print(f"  Data dir  : {data_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Device    : {device}")
    print(f"{'='*60}\n")

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Run the data preparation script first:")
        print(f"  cd data && python data.py --dataset {args.dataset} --appliance {args.appliance}")
        sys.exit(1)

    # Training
    if args.eval_only:
        if args.checkpoint is None:
            print("ERROR: --checkpoint is required when using --eval-only")
            sys.exit(1)
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = train(
            dataset=args.dataset,
            appliance=args.appliance,
            model_name=args.model,
            data_dir=data_dir,
            output_dir=output_dir,
            device=device,
        )

    # Evaluation
    evaluate(
        dataset=args.dataset,
        appliance=args.appliance,
        model_name=args.model,
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
    )


if __name__ == '__main__':
    main()
