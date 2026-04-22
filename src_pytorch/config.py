# Copyright (c) 2026 Sotirios Athanasoulias. MIT License — see LICENSE for details.
"""
Configuration for PyTorch NILM models.

This module centralises all static hyperparameters for models, training,
callbacks, and datasets. To run an experiment, only three values need to
be set externally: dataset name, appliance name, and model name.
All other parameters are resolved through the helper functions below.
"""

# =============================================================================
# MODEL CONFIGURATIONS
# Input window lengths and batch sizes are architecture-specific.
# TCN carries additional structural hyperparameters.
# =============================================================================
MODEL_CONFIGS = {
    # Seq2Point CNN: odd window so the target maps to the exact centre sample
    'cnn': {
        'input_window_length': 299,
        'batch_size': 1024,
        'training': {
            'epochs': 50,
            'learning_rate': 0.001,
            'early_stopping_patience': 5,
            'early_stopping_min_delta': 1e-6,
            'lr_scheduler': {
                'factor': 0.5,
                'patience': 3,
                'min_lr': 1e-6,
                'cooldown': 2,
            },
        },
    },
    # Seq2Seq CNN: same backbone as Seq2Point but predicts the full 299-sample window
    'cnn_seq2seq': {
        'input_window_length': 299,
        'batch_size': 512,
        'training': {
            'epochs': 15,
            'learning_rate': 0.0001,
            'early_stopping_patience': 5,
            'early_stopping_min_delta': 1e-6,
            'lr_scheduler': {
                'factor': 0.5,
                'patience': 3,
                'min_lr': 1e-6,
                'cooldown': 2,
            },
        },
    },
    # WaveNet TCN: entire window is predicted; dilated causal convolutions with gating
    'wavenet_tcn': {
        'input_window_length': 600,
        'batch_size': 128,
        'depth': 9,                                                  # Number of dilated conv blocks
        'nb_filters': [512, 360, 360, 256, 256, 256, 360, 360, 512],  # Filters per block
        'dropout': 0.1,                                              # Dropout rate
        'stacks': 1,                                                 # Number of TCN stacks
        'res_l2': 0,                                                 # L2 on residual connections
        'training': {
            'epochs': 100,
            'learning_rate': 0.001,
            'early_stopping_patience': 40,
            'early_stopping_min_delta': 1e-6,
            'lr_scheduler': {
                'factor': 0.5,
                'patience': 7,
                'min_lr': 1e-6,
                'cooldown': 2,
            },
        },
    },
}

# =============================================================================
# TRAINING PARAMETERS
# Adam optimiser settings and epoch budget shared across all experiments.
# =============================================================================
TRAINING = {
    'epochs': 50,             # Default max epochs (used by tcn; cnn/cnn_seq2seq override this)
    'learning_rate': 0.0001,  # Adam initial learning rate
    'beta_1': 0.9,            # Adam first moment decay
    'beta_2': 0.999,          # Adam second moment decay
    'epsilon': 1e-8,          # Adam numerical stability term
    'loss': 'mse',            # Loss function identifier
    'metrics': ['mse', 'mae'],
    # Early stopping — referenced directly in trainer setup
    'early_stopping_patience': 10,
    'early_stopping_min_delta': 1e-6,
    # ReduceLROnPlateau scheduler
    'lr_scheduler': {
        'factor': 0.5,      # Multiply LR by this factor on plateau
        'patience': 3,      # Epochs with no val_loss improvement before reducing LR
        'min_lr': 1e-6,     # Lower bound on the learning rate
        'cooldown': 2,      # Epochs to wait after a reduction before resuming normal operation
    },
}

# =============================================================================
# CALLBACKS
# Settings for early stopping, model checkpointing, and TensorBoard.
# =============================================================================
CALLBACKS = {
    'early_stopping': {
        'monitor': 'val_loss',
        'patience': 40,
        'min_delta': 1e-6,
        'mode': 'min',
        'verbose': True,
    },
    'model_checkpoint': {
        'monitor': 'val_loss',
        'mode': 'min',
        'save_best_only': True,   # Only write weights when val_loss improves
        'save_weights_only': True,
    },
    'tensorboard': {
        'update_freq': 'epoch',   # Log once per epoch
    },
}

# =============================================================================
# DATASET CONFIGURATIONS
# Raw data location relative to the data/ directory, sampling frequency,
# and upper clip value applied to aggregate power readings.
# =============================================================================
DATASET_CONFIGS = {
    'refit': {
        'location': 'CLEAN_REFIT_081116',
        'sampling': '8s',             # 8-second sampling interval
        'aggregate_cutoff': 10000,    # Clip aggregate above 10 kW
    },
    'plegma': {
        'location': 'PlegmaDataset_Clean/Clean_Dataset',
        'sampling': '10s',            # 10-second sampling interval
        'aggregate_cutoff': 10000,    # Clip aggregate above 10 kW
    },
}

# =============================================================================
# DATASET SPLITS
# House-based train / validation / test splits.
# Keeping the test house fully unseen ensures realistic generalisation testing.
# =============================================================================
DATASET_SPLITS = {
    'refit': {
        'dishwasher':      {'train': [5, 7, 9, 13, 16],             'val': [18], 'test': [20]},
        'washing_machine': {'train': [2, 5, 7, 9, 15, 16, 17],      'val': [18], 'test': [8]},
        'kettle':          {'train': [3, 4, 6, 7, 8, 9, 12, 13, 19, 20], 'val': [5], 'test': [2]},
        'microwave':       {'train': [10, 12, 19],                   'val': [17], 'test': [4]},
        'refrigerator':    {'train': [2, 5, 9],                      'val': [12], 'test': [15]},
    },
    'plegma': {
        # Houses 1–13 available; house 2 held out for testing across all appliances
        'ac_1':            {'train': [2, 3, 4, 6, 7, 8, 9, 11,10, 12, 13], 'val': [5], 'test': [1]},
        'boiler':          {'train': [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13], 'val': [10], 'test': [2]},
        'washing_machine': {'train': [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13], 'val': [10], 'test': [2]},
        'fridge':          {'train': [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13], 'val': [10], 'test': [2]},
    },
}

# =============================================================================
# APPLIANCE PARAMETERS
# Per-appliance threshold (ON/OFF classification boundary), cutoff (maximum
# rated power for normalisation), and aggregate normalisation statistics.
#
# Threshold  — power below this value (W) is treated as appliance OFF
# Cutoff     — max power (W) used for target normalisation: y / cutoff
# Mean / Std — aggregate z-score normalisation: (x - mean) / std
# =============================================================================

# REFIT dataset appliance parameters
REFIT_PARAMS = {
    'dishwasher': {
        'threshold': 10,
        'cutoff': 2500,
        'mean': 602.5479766992502,
        'std': 828.1060487606736,
    },
    'washing_machine': {
        'threshold': 20,
        'cutoff': 2500,
        'mean': 512.30344412,
        'std': 816.25221938,
    },
    'kettle': {
        'threshold': 2000,
        'cutoff': 3000,
        'mean': 500.1012921709225,
        'std': 749.2437943779291,
    },
    'microwave': {
        'threshold': 200,
        'cutoff': 1300,
        'mean': 489.55122864129567,
        'std': 696.0943467973808,
    },
    'refrigerator': {
        'threshold': 5,
        'cutoff': 1700,
        'mean': 600.1804355908282,
        'std': 944.5461745567851,
    },
}

# PLEGMA dataset appliance parameters
PLEGMA_PARAMS = {
    'ac_1': {
        'threshold': 50,
        'cutoff': 2300,
        'mean': 345.71,
        'std': 723.03,
        'min_on' : 100,   # minimum ON duration in samples (10-s intervals)
        'min_off': 50,    # minimum OFF gap in samples
    },
    'boiler': {
        'threshold': 800,
        'cutoff': 4000,
        'mean': 347.59,
        'std': 745.19,
        'min_on' : 30,
        'min_off': 6,
    },
    'washing_machine': {
        'threshold': 15,
        'cutoff': 2600,
        'mean': 344.4923608748837,
        'std': 731.6132455403795,
        'min_on'    : 2,
        'min_off'   : 100,
        'min_committed_duration': 250,
    },
    'fridge': {
        'threshold': 50,
        'cutoff': 400,
        'mean': 328.15979685575536,
        'std': 710.1635515683704,
        'min_on' : 10,
        'min_off': 2,
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_appliance_params(dataset_name: str, appliance_name: str) -> dict:
    """Return normalisation and threshold parameters for a given appliance.

    Args:
        dataset_name: ``'refit'`` or ``'plegma'``
        appliance_name: Appliance key, e.g. ``'boiler'``, ``'kettle'``

    Returns:
        Dict with keys ``threshold``, ``cutoff``, ``mean``, ``std``

    Raises:
        KeyError: If dataset or appliance name is not found
    """
    params_map = {'refit': REFIT_PARAMS, 'plegma': PLEGMA_PARAMS}
    return params_map[dataset_name][appliance_name]


def get_model_config(model_name: str) -> dict:
    """Return architecture hyperparameters for a given model.

    Args:
        model_name: ``'cnn'``, ``'gru'``, or ``'wavenet_tcn'``

    Returns:
        Dict containing at minimum ``input_window_length`` and ``batch_size``

    Raises:
        KeyError: If model name is not found
    """
    return MODEL_CONFIGS[model_name]


def get_dataset_config(dataset_name: str) -> dict:
    """Return dataset-level configuration.

    Args:
        dataset_name: ``'refit'`` or ``'plegma'``

    Returns:
        Dict with keys ``location``, ``sampling``, ``aggregate_cutoff``

    Raises:
        KeyError: If dataset name is not found
    """
    return DATASET_CONFIGS[dataset_name]


def get_training_config(model_name: str) -> dict:
    """Return training hyperparameters for a given model.

    Model-specific training config (defined inside MODEL_CONFIGS) takes
    priority. Falls back to the global TRAINING dict for models that do not
    define their own (e.g. gru, tcn).

    Args:
        model_name: ``'cnn'``, ``'cnn_seq2seq'``, or ``'wavenet_tcn'``

    Returns:
        Dict with keys ``epochs``, ``learning_rate``, ``early_stopping_patience``,
        ``early_stopping_min_delta``, and ``lr_scheduler``
    """
    return MODEL_CONFIGS.get(model_name, {}).get('training', TRAINING)


def get_dataset_split(dataset_name: str, appliance_name: str) -> dict:
    """Return the house-based train / val / test split for an appliance.

    Args:
        dataset_name: ``'refit'`` or ``'plegma'``
        appliance_name: Appliance key, e.g. ``'boiler'``

    Returns:
        Dict with keys ``train``, ``val``, ``test``, each a list of house IDs

    Raises:
        KeyError: If dataset or appliance name is not found
    """
    return DATASET_SPLITS[dataset_name][appliance_name]
