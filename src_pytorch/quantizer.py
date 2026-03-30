"""
src_pytorch/quantizer.py

TFLite full-integer INT8 quantization and Edge TPU compilation pipeline
for TCN NILM models.

Sections
--------
1. PyTorch Model Reconstruction  — rebuild pruned TCN from checkpoint
2. TF/Keras Model Building       — Keras Functional API TCN re-implementation
3. Weight Transfer               — PyTorch [out,in,k] → TF [k,in,out]
4. Validation                    — compare PyTorch vs TF predictions
5. TFLite Conversion             — INT8 full-integer quantization
6. Edge TPU Compilation          — edgetpu_compiler subprocess wrapper
7. TFLite Evaluation             — CPU interpreter inference + metrics
8. I/O Utilities                 — CSV / Excel helpers
"""

import itertools
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# TensorFlow / Keras — imported at module level because this module is
# exclusively for the TFLite quantization pipeline.  If TF is not installed,
# importing this module raises an ImportError with an actionable message.
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import (
        Activation, Concatenate, Conv1D, Dropout,
        LeakyReLU, Multiply, ReLU,
    )
except ImportError as exc:
    raise ImportError(
        "TensorFlow is required for src_pytorch.quantizer.\n"
        "Install it with:  pip install tensorflow"
    ) from exc

from .models import CNN_NILM, TCN_NILM, GRU_NILM
from .pruner import apply_torch_pruning
from .evaluator import compute_metrics, compute_status


# =============================================================================
# 1. PyTorch Model Reconstruction
# =============================================================================

def _load_tcn(cfg: dict, ckpt_path, device: torch.device) -> nn.Module:
    """Load a baseline (un-pruned) TCN_NILM from *ckpt_path*.

    Parameters
    ----------
    cfg      : dict          — must contain 'window', 'depth', 'filters',
                               'dropout', 'stacks' keys
    ckpt_path: str or Path   — path to the baseline checkpoint produced by
                               ``01_data_prep_training.ipynb``
    device   : torch.device

    Returns
    -------
    nn.Module — model in eval mode, loaded from checkpoint
    """
    model = TCN_NILM(
        input_window_length=cfg['window'],
        depth=cfg['depth'],
        nb_filters=cfg['filters'],
        dropout=cfg['dropout'],
        stacks=cfg['stacks'],
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device).eval()


def rebuild_pruned_tcn(
    cfg: dict,
    baseline_ckpt,
    pruning_ratio: float,
    finetuned_ckpt_path,
    device: torch.device,
) -> nn.Module:
    """Reproduce the pruned + fine-tuned TCN architecture deterministically.

    Pruning is *structural* (channels are removed), so the fine-tuned checkpoint
    has a different shape than the baseline.  To load it correctly we must:

    1. Build a fresh baseline TCN and load the original weights.
    2. Re-apply the same pruning (same magnitude ranking → same channels removed).
    3. Load the fine-tuned weights into the now-pruned architecture.

    .. note::
        The resulting model is intended for **weight export only** (to TF/Keras).
        It is always placed on CPU and set to eval mode.

    Parameters
    ----------
    cfg                 : dict          — TCN configuration dict with keys
                          'window', 'depth', 'filters', 'dropout', 'stacks',
                          'args_window_size'
    baseline_ckpt       : str or Path   — original (pre-pruning) checkpoint
    pruning_ratio       : float         — target *parameter* reduction fraction (e.g. 0.75);
                          passed directly to :func:`apply_torch_pruning`, which converts
                          it internally via :func:`param_ratio_to_channel_ratio`
    finetuned_ckpt_path : str or Path   — fine-tuned checkpoint from
                          ``03_pruning.ipynb``
    device              : torch.device

    Returns
    -------
    nn.Module — pruned TCN in eval mode with fine-tuned weights loaded

    Raises
    ------
    FileNotFoundError — if *finetuned_ckpt_path* does not exist
    """
    finetuned_ckpt_path = Path(finetuned_ckpt_path)
    if not finetuned_ckpt_path.exists():
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found:\n  {finetuned_ckpt_path}\n"
            "Run 03_pruning.ipynb first with matching settings."
        )

    model        = _load_tcn(cfg, baseline_ckpt, device)
    pruning_args = SimpleNamespace(window_size=cfg['args_window_size'])
    dummy        = torch.randn(1, cfg['window']).to(device)
    model, _     = apply_torch_pruning(model, pruning_args, dummy, pruning_ratio)
    model.load_state_dict(torch.load(finetuned_ckpt_path, map_location=device))
    return model.eval()


# =============================================================================
# 1b. CNN Model Reconstruction
# =============================================================================

def _load_cnn(cfg: dict, ckpt_path, device: torch.device) -> nn.Module:
    """Load a baseline (un-pruned) CNN_NILM from *ckpt_path*."""
    model = CNN_NILM(input_window_length=cfg['window'])
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device).eval()


def rebuild_pruned_cnn(
    cfg: dict,
    baseline_ckpt,
    pruning_ratio: float,
    finetuned_ckpt_path,
    device: torch.device,
) -> nn.Module:
    """Reproduce the pruned + fine-tuned CNN architecture deterministically.

    Follows the same three-step procedure as :func:`rebuild_pruned_tcn`:
    rebuild baseline → re-apply pruning → load fine-tuned weights.

    Parameters
    ----------
    cfg                 : dict — must contain 'window' and 'args_window_size'
    baseline_ckpt       : str or Path
    pruning_ratio       : float — target *parameter* reduction fraction (e.g. 0.75);
                          passed directly to :func:`apply_torch_pruning`, which converts
                          it internally via :func:`param_ratio_to_channel_ratio`
    finetuned_ckpt_path : str or Path
    device              : torch.device

    Returns
    -------
    nn.Module — pruned CNN_NILM in eval mode with fine-tuned weights
    """
    finetuned_ckpt_path = Path(finetuned_ckpt_path)
    if not finetuned_ckpt_path.exists():
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found:\n  {finetuned_ckpt_path}\n"
            "Run 03_pruning.ipynb first with matching settings."
        )

    model        = _load_cnn(cfg, baseline_ckpt, device)
    pruning_args = SimpleNamespace(window_size=cfg['args_window_size'])
    dummy        = torch.randn(1, cfg['window']).to(device)
    model, _     = apply_torch_pruning(model, pruning_args, dummy, pruning_ratio)
    model.load_state_dict(torch.load(finetuned_ckpt_path, map_location=device))
    return model.eval()


# =============================================================================
# 1c. GRU Model Reconstruction
# =============================================================================

def _load_gru(cfg: dict, ckpt_path, device: torch.device) -> nn.Module:
    """Load a baseline (un-pruned) GRU_NILM from *ckpt_path*."""
    model = GRU_NILM(input_window_length=cfg['window'])
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device).eval()


def rebuild_pruned_gru(
    cfg: dict,
    baseline_ckpt,
    pruning_ratio: float,
    finetuned_ckpt_path,
    device: torch.device,
) -> nn.Module:
    """Load the baseline GRU for quantization.

    GRU models cannot be structured-pruned with torch_pruning (bidirectional
    weight pairs block the entire channel dependency graph), so the baseline
    checkpoint is loaded directly — pruning_ratio and finetuned_ckpt_path are
    accepted for API consistency but are ignored.

    Parameters
    ----------
    cfg                 : dict — must contain 'window'
    baseline_ckpt       : str or Path
    pruning_ratio       : float — ignored for GRU
    finetuned_ckpt_path : str or Path — ignored for GRU
    device              : torch.device

    Returns
    -------
    nn.Module — baseline GRU_NILM in eval mode
    """
    print("GRU does not support structured pruning — loading baseline checkpoint.")
    return _load_gru(cfg, baseline_ckpt, device)


# =============================================================================
# 2. TF/Keras Model Building
# =============================================================================

def read_pruned_channels(pt_model: nn.Module, depth: int, stacks: int) -> tuple:
    """Extract the actual post-pruning channel counts from a PyTorch model.

    Pruning removes channels from Conv1d layers, so the true channel counts
    differ from the baseline configuration filters.  This function reads the
    weight shapes from the state dict to recover the real sizes.

    Parameters
    ----------
    pt_model : nn.Module — pruned TCN_NILM (must be loaded with fine-tuned ckpt)
    depth    : int       — number of gated blocks per stack (unchanged by pruning)
    stacks   : int       — number of stacks (unchanged by pruning)

    Returns
    -------
    initial_ch    : int       — output channels of the initial 1×1 conv
    block_filters : list[int] — output channels per gated block,
                                length = depth × stacks, ordered by block index
    """
    sd = pt_model.state_dict()
    initial_ch    = sd['initial_conv.weight'].shape[0]
    block_filters = [
        sd[f'gated_blocks.{i}.signal_conv.conv.weight'].shape[0]
        for i in range(depth * stacks)
    ]
    return initial_ch, block_filters


def build_tcn_keras(
    initial_ch: int,
    block_filters: list,
    depth: int,
    stacks: int,
    dropout_rate: float,
    seq_len: int,
) -> keras.Model:
    """Build a TF/Keras reimplementation of the pruned TCN_NILM.

    Uses the Keras Functional API to create a graph that mirrors the PyTorch
    architecture exactly, including WaveNet-style gated convolutions and skip
    connections.

    Architecture mapping
    --------------------
    +-------------------------------------+------------------------------------+
    | PyTorch                             | TF/Keras                           |
    +=====================================+====================================+
    | ``CausalConv1d(k=2, dil=2^i)``      | ``Conv1D(2, padding='causal',      |
    |                                     |   dilation_rate=2^i)``             |
    +-------------------------------------+------------------------------------+
    | ``F.relu(signal_conv(x))``          | ``ReLU()(signal_conv(x))``         |
    +-------------------------------------+------------------------------------+
    | ``torch.sigmoid(gate_conv(x))``     | ``Activation('sigmoid')(…)``       |
    +-------------------------------------+------------------------------------+
    | ``signal * gate``                   | ``Multiply()([signal, gate])``     |
    +-------------------------------------+------------------------------------+
    | ``torch.cat(skips, dim=1)``         | ``Concatenate(axis=-1)(skips)``    |
    +-------------------------------------+------------------------------------+
    | ``LeakyReLU(0.1)``                  | ``LeakyReLU(0.1)``                 |
    +-------------------------------------+------------------------------------+

    TF uses **channel-last** format ``(batch, seq_len, channels)`` throughout.

    Parameters
    ----------
    initial_ch    : int       — output channels of the initial 1×1 conv
                                (use :func:`read_pruned_channels`)
    block_filters : list[int] — output channels per gated block,
                                length = depth × stacks
    depth         : int       — gated blocks per stack
    stacks        : int       — number of stacks
    dropout_rate  : float     — dropout applied after each gated block
    seq_len       : int       — input sequence length (= window size)

    Returns
    -------
    keras.Model — uncompiled Functional API model
                  Input  shape : (batch, seq_len, 1)
                  Output shape : (batch, seq_len, 1)
    """
    inp = keras.Input(shape=(seq_len, 1), name='input')

    # Initial 1×1 conv — feature mixing, mirrors pt_model.initial_conv
    x = Conv1D(initial_ch, 1, padding='valid', use_bias=True,
               name='initial_conv')(inp)

    skip_connections = [x]

    for s in range(stacks):
        for i in range(depth):
            idx    = s * depth + i
            out_ch = block_filters[idx]
            dil    = 2 ** i

            # Signal path: causal dilated conv + ReLU
            sig = Conv1D(out_ch, 2, padding='causal', dilation_rate=dil,
                         use_bias=True, name=f'signal_conv_{s}_{i}')(x)
            sig = ReLU(name=f'signal_relu_{s}_{i}')(sig)

            # Gate path: causal dilated conv + Sigmoid
            gt  = Conv1D(out_ch, 2, padding='causal', dilation_rate=dil,
                         use_bias=True, name=f'gate_conv_{s}_{i}')(x)
            gt  = Activation('sigmoid', name=f'gate_sig_{s}_{i}')(gt)

            # Element-wise gating
            x = Multiply(name=f'mul_{s}_{i}')([sig, gt])

            if dropout_rate > 0:
                x = Dropout(dropout_rate, name=f'drop_{s}_{i}')(x)

            skip_connections.append(x)

    # Concatenate all skip connections along the channel axis (axis=-1 = channel-last)
    out = Concatenate(axis=-1, name='skip_cat')(skip_connections)

    # Final 1×1 projection + LeakyReLU
    out = Conv1D(1, 1, padding='valid', use_bias=True, name='final_conv')(out)
    out = LeakyReLU(0.1, name='leaky_relu')(out)

    return keras.Model(inputs=inp, outputs=out, name='TCN_NILM_TF')


# =============================================================================
# 2b. CNN TF/Keras Model Building
# =============================================================================

def read_pruned_cnn_channels(pt_model: nn.Module) -> tuple:
    """Extract actual post-pruning channel counts from a CNN_NILM.

    Reads the Sequential layer weight shapes from the state dict.
    The five Conv1d layers are at indices 0, 2, 4, 6, 9 in the Sequential.
    The intermediate Dense layer is at index 13.

    Parameters
    ----------
    pt_model : nn.Module — pruned CNN_NILM

    Returns
    -------
    filters     : list[int] — [ch1, ch2, ch3, ch4, ch5] output channels per Conv1d
    dense1_units: int       — output units of the first Dense layer (pruned from 1024)
    """
    sd = pt_model.state_dict()
    filters = [
        sd['network.0.weight'].shape[0],   # Conv1d 1: out_ch
        sd['network.2.weight'].shape[0],   # Conv1d 2: out_ch
        sd['network.4.weight'].shape[0],   # Conv1d 3: out_ch
        sd['network.6.weight'].shape[0],   # Conv1d 4: out_ch
        sd['network.9.weight'].shape[0],   # Conv1d 5: out_ch
    ]
    dense1_units = sd['network.13.weight'].shape[0]
    return filters, dense1_units


def build_cnn_keras(
    filters: list,
    seq_len: int,
    dense1_units: int = 1024,
) -> 'keras.Model':
    """Build a TF/Keras reimplementation of the pruned CNN_NILM.

    The CNN is a Seq2Point model: it takes a window of length *seq_len* and
    outputs a single scalar (the predicted power at the centre point).

    Architecture mapping (channel-last TF vs channel-first PyTorch)
    ---------------------------------------------------------------
    +------------------------------------------+---------------------------+
    | PyTorch                                  | TF/Keras                  |
    +==========================================+===========================+
    | ``Conv1d(k=10, no padding)``             | ``Conv1D(10, 'valid')``   |
    | ``Conv1d(k=8,  no padding)``             | ``Conv1D(8,  'valid')``   |
    | ``Conv1d(k=6,  no padding)``             | ``Conv1D(6,  'valid')``   |
    | ``Conv1d(k=5,  no padding)``             | ``Conv1D(5,  'valid')``   |
    | ``Conv1d(k=5,  no padding)``             | ``Conv1D(5,  'valid')``   |
    | ``Flatten``                              | ``Flatten``               |
    | ``Linear(ch5*(seq_len-29), 1024)``       | ``Dense(1024)``           |
    | ``Linear(1024, 1)``                      | ``Dense(1)``              |
    +------------------------------------------+---------------------------+

    Parameters
    ----------
    filters      : list[int] — [ch1, ch2, ch3, ch4, ch5] pruned output channels
    seq_len      : int       — input sequence length (= window size, default 299)
    dense1_units : int       — output units of the first Dense layer; use the value
                               returned by :func:`read_pruned_cnn_channels` so that
                               the Keras model matches the pruned PyTorch architecture
                               (default 1024 = un-pruned baseline)

    Returns
    -------
    keras.Model — Input shape: (batch, seq_len, 1)  Output shape: (batch, 1)
    """
    from tensorflow.keras.layers import Dense, Flatten, Permute

    ch1, ch2, ch3, ch4, ch5 = filters

    inp = keras.Input(shape=(seq_len, 1), name='input')

    x = Conv1D(ch1, 10, padding='valid', use_bias=True, name='conv1')(inp)
    x = ReLU(name='relu1')(x)
    x = Conv1D(ch2, 8, padding='valid', use_bias=True, name='conv2')(x)
    x = ReLU(name='relu2')(x)
    x = Conv1D(ch3, 6, padding='valid', use_bias=True, name='conv3')(x)
    x = ReLU(name='relu3')(x)
    x = Conv1D(ch4, 5, padding='valid', use_bias=True, name='conv4')(x)
    x = ReLU(name='relu4')(x)
    x = Dropout(0.2, name='drop4')(x)
    x = Conv1D(ch5, 5, padding='valid', use_bias=True, name='conv5')(x)
    x = ReLU(name='relu5')(x)
    x = Dropout(0.2, name='drop5')(x)
    # PyTorch Flatten operates on (batch, channels, length) → channels-first C-order.
    # TF Conv1D output is (batch, length, channels) → must transpose before flattening
    # so the Dense weight indices align with the PyTorch weight indices.
    x = Permute((2, 1), name='permute_flatten')(x)  # (batch, length, ch) → (batch, ch, length)
    x = Flatten(name='flatten')(x)
    x = Dense(dense1_units, use_bias=True, name='dense1')(x)
    x = ReLU(name='relu_dense')(x)
    x = Dropout(0.2, name='drop_dense')(x)
    x = Dense(1, use_bias=True, name='dense2')(x)

    return keras.Model(inputs=inp, outputs=x, name='CNN_NILM_TF')


# =============================================================================
# 2c. GRU TF/Keras Model Building
# =============================================================================

def read_pruned_gru_channels(pt_model: nn.Module) -> tuple:
    """Extract post-pruning channel/hidden-size counts from a GRU_NILM.

    Parameters
    ----------
    pt_model : nn.Module — pruned GRU_NILM

    Returns
    -------
    conv1_ch    : int — output channels of conv1 after pruning
    gru1_hidden : int — forward hidden size of gru1 (bidirectional output = 2×)
    gru2_hidden : int — forward hidden size of gru2
    fc1_out     : int — output units of fc1 after pruning
    """
    sd = pt_model.state_dict()
    conv1_ch    = sd['conv1.weight'].shape[0]        # (out_ch, 1, k)
    gru1_hidden = sd['gru1.weight_hh_l0'].shape[1]  # (3H, H) → H
    gru2_hidden = sd['gru2.weight_hh_l0'].shape[1]
    fc1_out     = sd['fc1.weight'].shape[0]          # (out, in)
    return conv1_ch, gru1_hidden, gru2_hidden, fc1_out


def build_gru_keras(
    conv1_ch: int,
    gru1_hidden: int,
    gru2_hidden: int,
    fc1_out: int,
    seq_len: int,
) -> 'keras.Model':
    """Build a TF/Keras reimplementation of the pruned GRU_NILM.

    Architecture mapping
    --------------------
    +-----------------------------------------------+------------------------------------------+
    | PyTorch                                       | TF/Keras                                 |
    +===============================================+==========================================+
    | ``Conv1d(1, ch, 4, padding='same')``          | ``Conv1D(ch, 4, padding='same')``        |
    | ``GRU(8, 32, bidirectional)``                 | ``Bidirectional(GRU(32, reset_after=F))``|
    | ``x[:, -1, :]`` (last time step)              | ``GRU(..., return_sequences=False)``     |
    | ``Linear(128, 64)``                           | ``Dense(64)``                            |
    +-----------------------------------------------+------------------------------------------+

    ``reset_after=False`` matches PyTorch's GRU formulation (reset gate applied
    before the recurrent matrix multiplication).

    Parameters
    ----------
    conv1_ch    : int — output channels of conv1 (use :func:`read_pruned_gru_channels`)
    gru1_hidden : int — forward hidden size of first BiGRU
    gru2_hidden : int — forward hidden size of second BiGRU
    fc1_out     : int — output units of fc1
    seq_len     : int — input sequence length (= window size, default 199)

    Returns
    -------
    keras.Model — Input shape: (batch, seq_len, 1)  Output shape: (batch, 1)
    """
    from tensorflow.keras.layers import Bidirectional
    from tensorflow.keras.layers import GRU as KerasGRU
    from tensorflow.keras.layers import Dense

    inp = keras.Input(shape=(seq_len, 1), name='input')

    x = Conv1D(conv1_ch, 4, padding='same', use_bias=True, name='conv1')(inp)
    x = ReLU(name='relu_conv')(x)

    # First Bidirectional GRU — return_sequences=True so gru2 sees full sequence
    x = Bidirectional(
        KerasGRU(gru1_hidden, return_sequences=True, reset_after=False, name='gru1_cell'),
        name='bigru1',
    )(x)
    x = Dropout(0.5, name='drop1')(x)

    # Second Bidirectional GRU — return_sequences=False → last time step only
    x = Bidirectional(
        KerasGRU(gru2_hidden, return_sequences=False, reset_after=False, name='gru2_cell'),
        name='bigru2',
    )(x)
    x = Dropout(0.5, name='drop2')(x)

    x = Dense(fc1_out, use_bias=True, name='fc1')(x)
    x = ReLU(name='relu_dense')(x)
    x = Dropout(0.5, name='drop3')(x)
    x = Dense(1, use_bias=True, name='fc2')(x)

    return keras.Model(inputs=inp, outputs=x, name='GRU_NILM_TF')


# =============================================================================
# 3. Weight Transfer
# =============================================================================

def transfer_weights(
    tf_model: keras.Model,
    pt_state_dict: dict,
    depth: int,
    stacks: int,
) -> None:
    """Copy weights from a pruned PyTorch state dict into a TF/Keras model.

    PyTorch ``Conv1d`` weights are stored as ``[out_ch, in_ch, kernel]``.
    TF/Keras ``Conv1D`` weights are stored as ``[kernel, in_ch, out_ch]``.
    The required transformation is::

        tf_weight = pt_weight.permute(2, 1, 0).numpy()

    The TF model must have been built by :func:`build_tcn_keras` with layer
    names that match the expected pattern
    (``initial_conv``, ``signal_conv_{s}_{i}``, ``gate_conv_{s}_{i}``,
    ``final_conv``).

    Parameters
    ----------
    tf_model       : keras.Model — target model (weights set in-place)
    pt_state_dict  : dict        — ``pt_model.state_dict()``
    depth          : int         — gated blocks per stack
    stacks         : int         — number of stacks
    """
    def _set_conv(layer_name: str, pt_w_key: str, pt_b_key: str) -> None:
        w = pt_state_dict[pt_w_key].cpu().permute(2, 1, 0).numpy()
        b = pt_state_dict[pt_b_key].cpu().numpy()
        tf_model.get_layer(layer_name).set_weights([w, b])

    _set_conv('initial_conv', 'initial_conv.weight', 'initial_conv.bias')

    for s in range(stacks):
        for i in range(depth):
            idx = s * depth + i
            _set_conv(
                f'signal_conv_{s}_{i}',
                f'gated_blocks.{idx}.signal_conv.conv.weight',
                f'gated_blocks.{idx}.signal_conv.conv.bias',
            )
            _set_conv(
                f'gate_conv_{s}_{i}',
                f'gated_blocks.{idx}.gate_conv.conv.weight',
                f'gated_blocks.{idx}.gate_conv.conv.bias',
            )

    _set_conv('final_conv', 'final_conv.weight', 'final_conv.bias')
    print('Weights transferred successfully.')


# =============================================================================
# 3b. CNN Weight Transfer
# =============================================================================

def transfer_cnn_weights(
    tf_model: 'keras.Model',
    pt_state_dict: dict,
) -> None:
    """Copy weights from a pruned CNN PyTorch state dict into a TF/Keras model.

    Conv1d:  ``pt[out, in, k]``  →  ``tf[k, in, out]``  (permute 2,1,0)
    Linear:  ``pt[out, in]``     →  ``tf[in, out]``      (transpose)

    Parameters
    ----------
    tf_model      : keras.Model — must have been built by :func:`build_cnn_keras`
    pt_state_dict : dict        — ``pt_model.state_dict()``
    """
    def _set_conv(layer_name, pt_w_key, pt_b_key):
        w = pt_state_dict[pt_w_key].cpu().permute(2, 1, 0).numpy()
        b = pt_state_dict[pt_b_key].cpu().numpy()
        tf_model.get_layer(layer_name).set_weights([w, b])

    def _set_dense(layer_name, pt_w_key, pt_b_key):
        w = pt_state_dict[pt_w_key].cpu().numpy().T   # (out, in) → (in, out)
        b = pt_state_dict[pt_b_key].cpu().numpy()
        tf_model.get_layer(layer_name).set_weights([w, b])

    _set_conv('conv1', 'network.0.weight', 'network.0.bias')
    _set_conv('conv2', 'network.2.weight', 'network.2.bias')
    _set_conv('conv3', 'network.4.weight', 'network.4.bias')
    _set_conv('conv4', 'network.6.weight', 'network.6.bias')
    _set_conv('conv5', 'network.9.weight', 'network.9.bias')
    _set_dense('dense1', 'network.13.weight', 'network.13.bias')
    _set_dense('dense2', 'network.16.weight', 'network.16.bias')

    print('CNN weights transferred successfully.')


# =============================================================================
# 3c. GRU Weight Transfer
# =============================================================================

def _pt_to_tf_gru_layer(pt_ih, pt_hh, pt_ih_bias, pt_hh_bias):
    """Convert one direction's PyTorch GRU weights to TF Keras (reset_after=False).

    PyTorch gate order : [r (0:H), z (H:2H), n (2H:3H)]
    TF Keras gate order: [z (0:H), r (H:2H), h (2H:3H)]

    TF bias = pt_bias_ih + pt_bias_hh  (combined for reset_after=False)

    Returns
    -------
    kernel           : np.ndarray (input_size, 3*H)
    recurrent_kernel : np.ndarray (H, 3*H)
    bias             : np.ndarray (3*H,)
    """
    H   = pt_ih.shape[0] // 3
    idx = list(range(H, 2 * H)) + list(range(H)) + list(range(2 * H, 3 * H))

    kernel           = pt_ih.cpu().numpy()[idx, :].T
    recurrent_kernel = pt_hh.cpu().numpy()[idx, :].T
    bias             = (pt_ih_bias.cpu().numpy() + pt_hh_bias.cpu().numpy())[idx]
    return kernel, recurrent_kernel, bias


def transfer_gru_weights(tf_model: 'keras.Model', pt_state_dict: dict) -> None:
    """Copy weights from a pruned GRU PyTorch state dict into a TF/Keras model.

    Handles:
    - Conv1d  : ``pt[out, in, k]``  → ``tf[k, in, out]``
    - BiGRU   : gate reorder [r,z,n] → [z,r,h] + bias combination
    - Linear  : ``pt[out, in]``     → ``tf[in, out]``

    Parameters
    ----------
    tf_model      : keras.Model — must have been built by :func:`build_gru_keras`
    pt_state_dict : dict        — ``pt_model.state_dict()``
    """
    sd = pt_state_dict

    # Conv1d: pt [out, in, k] → tf [k, in, out]
    tf_model.get_layer('conv1').set_weights([
        sd['conv1.weight'].cpu().permute(2, 1, 0).numpy(),
        sd['conv1.bias'].cpu().numpy(),
    ])

    # BiGRU 1 — forward and backward directions
    fw1_k, fw1_rk, fw1_b = _pt_to_tf_gru_layer(
        sd['gru1.weight_ih_l0'],         sd['gru1.weight_hh_l0'],
        sd['gru1.bias_ih_l0'],           sd['gru1.bias_hh_l0'],
    )
    bw1_k, bw1_rk, bw1_b = _pt_to_tf_gru_layer(
        sd['gru1.weight_ih_l0_reverse'], sd['gru1.weight_hh_l0_reverse'],
        sd['gru1.bias_ih_l0_reverse'],   sd['gru1.bias_hh_l0_reverse'],
    )
    tf_model.get_layer('bigru1').set_weights(
        [fw1_k, fw1_rk, fw1_b, bw1_k, bw1_rk, bw1_b]
    )

    # BiGRU 2 — forward and backward directions
    fw2_k, fw2_rk, fw2_b = _pt_to_tf_gru_layer(
        sd['gru2.weight_ih_l0'],         sd['gru2.weight_hh_l0'],
        sd['gru2.bias_ih_l0'],           sd['gru2.bias_hh_l0'],
    )
    bw2_k, bw2_rk, bw2_b = _pt_to_tf_gru_layer(
        sd['gru2.weight_ih_l0_reverse'], sd['gru2.weight_hh_l0_reverse'],
        sd['gru2.bias_ih_l0_reverse'],   sd['gru2.bias_hh_l0_reverse'],
    )
    tf_model.get_layer('bigru2').set_weights(
        [fw2_k, fw2_rk, fw2_b, bw2_k, bw2_rk, bw2_b]
    )

    # Dense layers: pt [out, in] → tf [in, out]
    tf_model.get_layer('fc1').set_weights([
        sd['fc1.weight'].cpu().numpy().T, sd['fc1.bias'].cpu().numpy(),
    ])
    tf_model.get_layer('fc2').set_weights([
        sd['fc2.weight'].cpu().numpy().T, sd['fc2.bias'].cpu().numpy(),
    ])

    print('GRU weights transferred successfully.')


# =============================================================================
# 4. Validation
# =============================================================================

def validate_weight_transfer(
    pt_model: nn.Module,
    tf_model: keras.Model,
    data_loader,
    device: torch.device,
    tolerance: float = 1e-3,
) -> dict:
    """Compare predictions from the PyTorch and TF/Keras models on one test batch.

    A mean absolute difference below *tolerance* (in normalised units) confirms
    that :func:`transfer_weights` mapped the parameters correctly.

    Parameters
    ----------
    pt_model    : nn.Module  — pruned PyTorch TCN (eval mode)
    tf_model    : keras.Model
    data_loader             — must expose ``.test`` (a DataLoader)
    device      : torch.device
    tolerance   : float      — maximum acceptable mean |PT − TF| (default 1e-3)

    Returns
    -------
    dict with keys:
        mae_diff : float — mean absolute difference (normalised)
        max_diff : float — max absolute difference (normalised)
        passed   : bool  — True if mae_diff < tolerance
    """
    test_batch_x, _ = next(iter(data_loader.test))
    x_np = test_batch_x.cpu().numpy()
    # CNN DataLoader returns (batch, window); TF model always expects (batch, window, 1)
    if x_np.ndim == 2:
        x_np = x_np[..., np.newaxis]

    pt_model.eval()
    with torch.no_grad():
        pt_out = pt_model(test_batch_x.to(device)).cpu().numpy().flatten()

    tf_out = tf_model(x_np, training=False).numpy().flatten()

    mae_diff = float(np.mean(np.abs(pt_out - tf_out)))
    max_diff = float(np.max(np.abs(pt_out - tf_out)))
    passed   = mae_diff < tolerance

    status = 'PASS' if passed else 'WARN — check weight mapping'
    print(f'  Mean |PT − TF|  : {mae_diff:.8f}  (normalised units)')
    print(f'  Max  |PT − TF|  : {max_diff:.8f}  (normalised units)')
    print(f'  Validation      : {status}  (threshold = {tolerance})')

    return {'mae_diff': mae_diff, 'max_diff': max_diff, 'passed': passed}


# =============================================================================
# 5. TFLite Conversion
# =============================================================================

def convert_to_tflite_int8(
    tf_model: keras.Model,
    data_loader,
    window: int,
    n_calib_batches: int,
    out_path,
) -> Path:
    """Convert a TF/Keras model to TFLite full-integer INT8.

    Uses a representative dataset from the training split to calibrate
    activation quantization ranges.  Falls back to adding
    ``SELECT_TF_OPS`` if the strict ``TFLITE_BUILTINS_INT8``-only
    conversion fails (e.g. due to unsupported ops such as LeakyReLU).

    Parameters
    ----------
    tf_model        : keras.Model
    data_loader                  — must expose ``.train`` (a DataLoader)
                                   yielding ``(batch_x, _)`` where
                                   ``batch_x`` shape is ``(batch, window, 1)``
    window          : int        — input sequence length
    n_calib_batches : int        — number of training batches used for
                                   calibration
    out_path        : str or Path — destination ``.tflite`` file path

    Returns
    -------
    Path — *out_path* (guaranteed to exist after successful conversion)
    """
    out_path = Path(out_path)

    def _representative_dataset():
        for x, _ in tqdm(
            itertools.islice(data_loader.train, n_calib_batches),
            total=n_calib_batches,
            desc='Calibrating',
            unit='batch',
        ):
            for sample in x.cpu().numpy():
                yield [sample.reshape(1, window, 1).astype(np.float32)]

    def _make_converter():
        c = tf.lite.TFLiteConverter.from_keras_model(tf_model)
        c.optimizations          = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = _representative_dataset
        return c

    print(f'Converting to TFLite INT8 (calibrating with {n_calib_batches} batches)...')

    # Attempt 1 — strict full-integer INT8
    converter = _make_converter()
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type      = tf.int8
    converter.inference_output_type     = tf.int8
    try:
        tflite_bytes = converter.convert()
        print('Full-integer INT8 conversion succeeded.')
    except Exception as exc:
        # Attempt 2 — add SELECT_TF_OPS for unsupported ops (e.g. LeakyReLU)
        print(f'Full INT8 failed ({exc}).\nFalling back to TFLITE_BUILTINS_INT8 + SELECT_TF_OPS...')
        converter2 = _make_converter()
        converter2.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter2.inference_input_type  = tf.int8
        converter2.inference_output_type = tf.int8
        tflite_bytes = converter2.convert()
        print('Conversion with SELECT_TF_OPS fallback succeeded.')

    out_path.write_bytes(tflite_bytes)
    size_mb = round(out_path.stat().st_size / (1024 ** 2), 3)
    print(f'\nTFLite INT8 model saved : {out_path}')
    print(f'File size               : {size_mb:.3f} MB')
    return out_path


# =============================================================================
# 6. Edge TPU Compilation
# =============================================================================

def compile_edgetpu(tflite_path, out_dir, timeout: int = 180) -> Path | None:
    """Compile a ``.tflite`` model for the Google Coral Edge TPU.

    Runs ``edgetpu_compiler`` as a subprocess and prints the op-coverage
    report.  The compiled ``_edgetpu.tflite`` file is written to *out_dir*.

    If ``edgetpu_compiler`` is not installed (``FileNotFoundError``) or the
    process times out, a warning is printed and ``None`` is returned —
    the TFLite INT8 model remains usable for CPU evaluation.

    Parameters
    ----------
    tflite_path : str or Path — path to the source ``.tflite`` file
    out_dir     : str or Path — directory where the compiler writes its output
    timeout     : int         — maximum seconds to wait (default 180)

    Returns
    -------
    Path | None — path to the ``_edgetpu.tflite`` file, or ``None`` if
                  compilation was skipped / failed
    """
    tflite_path = Path(tflite_path)
    out_dir     = Path(out_dir)

    stem         = tflite_path.stem           # e.g. 'tcn_boiler_pruned_50pct_int8'
    edgetpu_name = stem + '_edgetpu.tflite'
    edgetpu_path = out_dir / edgetpu_name

    try:
        result = subprocess.run(
            ['edgetpu_compiler', '--out_dir', str(out_dir), str(tflite_path)],
            capture_output=True, text=True, timeout=timeout,
        )
        print('=== Edge TPU Compiler Output ===')
        print(result.stdout)
        if result.stderr:
            print('STDERR:', result.stderr)

        if edgetpu_path.exists():
            print(f'Edge TPU model saved : {edgetpu_path}')
            print(f'Size                 : {edgetpu_path.stat().st_size / 1024:.1f} KB')
            return edgetpu_path
        else:
            print('\nNote: compiler ran but output file not found at expected path.')
            print('Check the compiler output above for the actual output location.')
            return None

    except FileNotFoundError:
        print('edgetpu_compiler not found in PATH.')
        print('Install from https://coral.ai/docs/edgetpu/compiler/')
        print('\nSkipping compilation — the TFLite INT8 model can still be evaluated on CPU.')
        return None

    except subprocess.TimeoutExpired:
        print(f'edgetpu_compiler timed out after {timeout} s.')
        return None


# =============================================================================
# 7. TFLite Evaluation
# =============================================================================

def evaluate_tflite(
    tflite_path,
    data_loader,
    window: int,
    cutoff: float,
    threshold: float,
    min_on: int = None,
    min_off: int = None,
    max_length: int = None,
) -> tuple:
    """Evaluate a TFLite INT8 model on the test split using the CPU interpreter.

    For each test sample the function:

    1. **Quantizes** the float32 input to int8 using the tensor's scale / zero-point.
    2. **Invokes** the TFLite interpreter.
    3. **Dequantizes** the int8 output back to float32.

    The raw predictions are aligned to the ground-truth array (TCN Seq2Seq:
    take the first ``len(preds)`` labels), denormalised by *cutoff*, and clipped
    to ``[threshold, cutoff]``.

    Parameters
    ----------
    tflite_path : str or Path — path to the ``.tflite`` file
    data_loader              — must expose ``.test`` (DataLoader) and
                               ``.test_labels`` (1-D numpy array, normalised)
    window      : int        — input sequence length
    cutoff      : float      — appliance power normalisation ceiling in Watts
    threshold   : float      — ON/OFF boundary in Watts
    min_on      : int        — minimum ON-duration for status computation
                               (enables ``f1_complex`` when provided)
    min_off     : int        — minimum OFF-duration for status computation
    max_length  : int        — optional stricter final length filter passed to
                               :func:`~src_pytorch.evaluator.compute_status`

    Returns
    -------
    (metrics, gt, pred, gt_status, pred_status)
        metrics    : dict — see :func:`~src_pytorch.pruner.compute_metrics`
        gt         : np.ndarray — denormalised ground truth in Watts
        pred       : np.ndarray — denormalised predictions in Watts (clipped)
        gt_status  : np.ndarray or None — binary ON/OFF status for ground truth
        pred_status: np.ndarray or None — binary ON/OFF status for predictions
    """
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_det  = interpreter.get_input_details()[0]
    output_det = interpreter.get_output_details()[0]

    in_scale,  in_zp  = input_det['quantization']
    out_scale, out_zp = output_det['quantization']

    print(f'  Input  dtype / shape : {input_det["dtype"].__name__} {input_det["shape"]}')
    print(f'  Output dtype / shape : {output_det["dtype"].__name__} {output_det["shape"]}')
    print(f'  Input  quant params  : scale={in_scale:.6f}, zero_point={in_zp}')
    print(f'  Output quant params  : scale={out_scale:.6f}, zero_point={out_zp}')

    all_preds = []
    all_gt    = []
    for batch_x, batch_y in tqdm(data_loader.test, desc='TFLite inference'):
        for sample, label in zip(batch_x.cpu().numpy(), batch_y.cpu().numpy()):
            x = sample.reshape(1, window, 1).astype(np.float32)

            # float32 → int8
            x_q = np.clip(np.round(x / in_scale + in_zp), -128, 127).astype(np.int8)
            interpreter.set_tensor(input_det['index'], x_q)
            interpreter.invoke()

            # int8 → float32
            out_q = interpreter.get_tensor(output_det['index'])
            out_f = (out_q.astype(np.float32) - out_zp) * out_scale
            all_preds.append(out_f.flatten())
            all_gt.append(label.flatten())

    preds_norm = np.concatenate(all_preds)
    # Collect gt from batch_y (works for CNN Seq2Point and TCN Seq2Seq alike)
    gt_norm = np.concatenate(all_gt)

    gt   = gt_norm    * cutoff
    pred = preds_norm * cutoff

    pred[pred < threshold] = 0
    pred[pred > cutoff]    = cutoff

    metrics = compute_metrics(gt, pred, threshold, min_on=min_on, min_off=min_off, max_length=max_length)

    gt_status   = None
    pred_status = None
    if min_on is not None and min_off is not None:
        gt_status   = compute_status(gt,   threshold, min_on, min_off, max_length)
        pred_status = compute_status(pred, threshold, min_on, min_off, max_length)

    return metrics, gt, pred, gt_status, pred_status


# =============================================================================
# 8. I/O Utilities
# =============================================================================

def save_predictions_csv(
    gt: np.ndarray,
    pred: np.ndarray,
    csv_path,
    gt_status: np.ndarray = None,
    pred_status: np.ndarray = None,
) -> Path:
    """Save ground-truth and prediction arrays to a CSV file.

    Parameters
    ----------
    gt          : np.ndarray — ground-truth power values in Watts
    pred        : np.ndarray — predicted power values in Watts
    csv_path    : str or Path — destination file path
    gt_status   : np.ndarray or None — binary ON/OFF status for ground truth
    pred_status : np.ndarray or None — binary ON/OFF status for predictions

    Returns
    -------
    Path — *csv_path*
    """
    csv_path = Path(csv_path)
    data = {'ground_truth_W': gt, 'prediction_W': pred}
    if gt_status is not None and pred_status is not None:
        data['ground_truth_status'] = gt_status.astype(int)
        data['predicted_status']    = pred_status.astype(int)
    pd.DataFrame(data).to_csv(csv_path, index=False)
    print(f'Predictions saved : {csv_path.name}')
    return csv_path


def upsert_excel_row(row_dict: dict, excel_path) -> None:
    """Append or update a single row in a comparative-results Excel file.

    If *excel_path* already exists, the row whose ``'Model'`` value matches
    ``row_dict['Model']`` is replaced; all other rows are preserved.
    If the file does not exist it is created.

    Parameters
    ----------
    row_dict   : dict        — column → value mapping for the new / updated row;
                               must contain a ``'Model'`` key used as primary key
    excel_path : str or Path — path to the ``.xlsx`` file
    """
    excel_path = Path(excel_path)

    if excel_path.exists():
        existing_df = pd.read_excel(excel_path)
        mask        = existing_df['Model'] != row_dict['Model']
        updated_df  = pd.concat(
            [existing_df[mask], pd.DataFrame([row_dict])], ignore_index=True
        )
        print(f'Upserting row in existing Excel: {excel_path}')
    else:
        updated_df = pd.DataFrame([row_dict])
        print(f'Creating new Excel: {excel_path}')

    updated_df.to_excel(excel_path, index=False)
    print(f'Excel updated: {excel_path}')
