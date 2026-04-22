# Copyright (c) 2026 Sotirios Athanasoulias. MIT License — see LICENSE for details.
"""
PyTorch implementation of OpenNILM

This package provides PyTorch implementations of NILM (Non-Intrusive Load Monitoring)
models including CNN and TCN architectures.
"""

from .models import CNN_NILM, TCN_NILM, CNN_NILM_Seq2Seq, get_model
from .data_loader import NILMDataset, DataLoaderNILM, SimpleNILMDataLoader
from .trainer import Trainer, train_model, EarlyStopping, ModelCheckpoint, TrainingHistory
from .tester import SimpleTester, load_model
from .config import (
    MODEL_CONFIGS,
    TRAINING,
    CALLBACKS,
    DATASET_CONFIGS,
    DATASET_SPLITS,
    REFIT_PARAMS,
    PLEGMA_PARAMS,
    get_appliance_params,
    get_model_config,
    get_dataset_config,
    get_dataset_split
)
from .utils import (
    set_seeds,
    create_experiment_directories,
    get_device,
    count_parameters,
    print_model_summary,
    save_checkpoint,
    load_checkpoint
)


from .evaluator import (
    run_predictions,
    compute_status,
    compute_metrics,
    evaluate_model,
)
from .pruner import (
    count_ops_and_params,
    count_parameters_per_layer,
    get_model_stats,
    get_model_sparsity,
    apply_torch_pruning,
    apply_unstructured_pruning,
    remove_pruning_masks,
    get_prunable_parameters,
    param_ratio_to_channel_ratio,
)
from .pipeline import (
    build_nilm_model,
    get_data_loader,
    run_training,
    run_evaluation,
    save_pipeline_results,
)

# quantizer is imported lazily (TF may not be installed in every environment)
# Users who need it should import directly:
#   from src_pytorch.quantizer import rebuild_pruned_tcn, build_tcn_keras, ...

__version__ = '1.0.0'
__all__ = [
    # Models
    'CNN_NILM',
    'TCN_NILM',
    'CNN_NILM_Seq2Seq',
    'get_model',

    # Data
    'NILMDataset',
    'DataLoaderNILM',
    'SimpleNILMDataLoader',

    # Training
    'Trainer',
    'train_model',
    'EarlyStopping',
    'ModelCheckpoint',
    'TrainingHistory',

    # Testing
    'SimpleTester',
    'load_model',

    # Config
    'MODEL_CONFIGS',
    'TRAINING',
    'CALLBACKS',
    'DATASET_CONFIGS',
    'DATASET_SPLITS',
    'REFIT_PARAMS',
    'PLEGMA_PARAMS',
    'get_appliance_params',
    'get_model_config',
    'get_dataset_config',
    'get_dataset_split',

    # Utils
    'set_seeds',
    'create_experiment_directories',
    'get_device',
    'count_parameters',
    'print_model_summary',
    'save_checkpoint',
    'load_checkpoint',

    # Evaluation
    'run_predictions',
    'compute_status',
    'compute_metrics',
    'evaluate_model',

    # Pruner
    'count_ops_and_params',
    'count_parameters_per_layer',
    'get_model_stats',
    'get_model_sparsity',
    'apply_torch_pruning',
    'apply_unstructured_pruning',
    'remove_pruning_masks',
    'get_prunable_parameters',
    'param_ratio_to_channel_ratio',

    # Pipeline
    'build_nilm_model',
    'get_data_loader',
    'run_training',
    'run_evaluation',
    'save_pipeline_results',
    # run_quantization intentionally omitted from __all__ (lazy TF import)
]
