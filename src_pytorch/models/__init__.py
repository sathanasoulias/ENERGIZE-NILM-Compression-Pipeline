# Copyright (c) 2026 Sotirios Athanasoulias. MIT License — see LICENSE for details.
"""
PyTorch Models for NILM
"""

from .cnn import CNN_NILM, get_model as get_cnn_model
from .tcn import TCN_NILM, get_model as get_tcn_model
from .cnn_seq2seq import CNN_NILM_Seq2Seq, get_model as get_cnn_seq2seq_model


def get_model(model_name: str, **kwargs):
    """
    Factory function to get a model by name.

    Args:
        model_name: Name of the model ('cnn', 'cnn_seq2seq', 'tcn')
        **kwargs: Model-specific arguments

    Returns:
        Instantiated model
    """
    models = {
        'cnn': get_cnn_model,
        'wavenet_tcn': get_tcn_model,
        'cnn_seq2seq': get_cnn_seq2seq_model,
    }

    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    return models[model_name.lower()](**kwargs)


__all__ = [
    'CNN_NILM', 'TCN_NILM', 'CNN_NILM_Seq2Seq',
    'get_model', 'get_cnn_model', 'get_tcn_model', 'get_cnn_seq2seq_model'
]
