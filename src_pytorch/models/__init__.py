"""
PyTorch Models for NILM
"""

from .cnn import CNN_NILM, get_model as get_cnn_model
from .gru import GRU_NILM, get_model as get_gru_model
from .tcn import TCN_NILM, get_model as get_tcn_model


def get_model(model_name: str, **kwargs):
    """
    Factory function to get a model by name.

    Args:
        model_name: Name of the model ('cnn', 'gru', 'tcn')
        **kwargs: Model-specific arguments

    Returns:
        Instantiated model
    """
    models = {
        'cnn': get_cnn_model,
        'gru': get_gru_model,
        'tcn': get_tcn_model
    }

    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    return models[model_name.lower()](**kwargs)


__all__ = [
    'CNN_NILM', 'GRU_NILM', 'TCN_NILM',
    'get_model', 'get_cnn_model', 'get_gru_model', 'get_tcn_model'
]
