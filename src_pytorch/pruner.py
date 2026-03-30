"""
src_pytorch/pruner.py

Model-cost statistics and pruning utilities for NILM models (CNN, GRU, TCN).

Sections
--------
1. Model Statistics     — parameter counts, MACs, memory footprint
2. Structured Pruning   — magnitude-based global channel pruning (torch_pruning)
3. Unstructured Pruning — global L1 weight pruning (torch.nn.utils.prune)
                          Works with CNN, GRU, and TCN.

Inference, metrics, and evaluation utilities have been moved to
src_pytorch/evaluator.py.
"""

import torch
import torch.nn as nn

# torch_pruning is imported lazily inside the functions that need it so that
# the rest of src_pytorch remains importable even when the package is not
# installed (e.g. during plain training/evaluation runs).


# =============================================================================
# 1. Model Statistics
# =============================================================================

def count_ops_and_params(model: nn.Module, inputs: torch.Tensor):
    """Return (MACs, parameter_count) for *model* given *inputs*.

    Parameters
    ----------
    model  : nn.Module       — the model to profile
    inputs : torch.Tensor    — a representative dummy input tensor

    Returns
    -------
    macs   : int — multiply-accumulate operations
    params : int — total trainable parameter count
    """
    import torch_pruning as tp
    macs, params = tp.utils.count_ops_and_params(model, inputs)
    return macs, params


def count_parameters_per_layer(model: nn.Module) -> dict:
    """Return a dict mapping layer name → trainable parameter count.

    Only Conv1d, Conv2d and Linear layers are included.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    dict[str, int]
    """
    layer_params = {}
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
            layer_params[name] = params
    return layer_params


def get_model_stats(model: nn.Module, dummy_input: torch.Tensor) -> tuple:
    """Return (params, MACs, size_MB) for *model*.

    Parameters
    ----------
    model       : nn.Module
    dummy_input : torch.Tensor — a representative single-sample input

    Returns
    -------
    params : int   — total trainable parameters
    macs   : int   — multiply-accumulate operations
    mb     : float — model weight memory footprint in megabytes
    """
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs, _ = count_ops_and_params(model, dummy_input)
    mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    return params, macs, round(mb, 3)


# =============================================================================
# 2. Structured Pruning
# =============================================================================

def param_ratio_to_channel_ratio(target_param_ratio: float) -> float:
    """Convert a user-facing parameter-reduction ratio to a channel pruning ratio.

    ``torch_pruning``'s ``MetaPruner`` accepts a *channel* pruning ratio ``p``,
    meaning it removes a fraction ``p`` of both the input and output channels of
    each layer.  Because both dimensions are scaled, the effective parameter
    reduction is approximately:

        parameter_reduction ≈ 1 − (1 − p)²

    Inverting this relationship gives the channel ratio required to achieve a
    desired parameter reduction:

        p = 1 − √(1 − target_param_ratio)

    Examples
    --------
    >>> param_ratio_to_channel_ratio(0.50)   # remove ~50 % of params → p ≈ 0.293
    >>> param_ratio_to_channel_ratio(0.75)   # remove ~75 % of params → p = 0.50

    Parameters
    ----------
    target_param_ratio : float — desired fraction of parameters to remove,
                                 must be in the open interval (0, 1)

    Returns
    -------
    float — channel pruning ratio to pass to :func:`apply_torch_pruning`

    Raises
    ------
    ValueError — if *target_param_ratio* is not in (0, 1)
    """
    import math
    if not 0.0 < target_param_ratio < 1.0:
        raise ValueError(
            f"target_param_ratio must be in (0, 1), got {target_param_ratio}"
        )
    return 1.0 - math.sqrt(1.0 - target_param_ratio)
    
 

def apply_torch_pruning(
    model: nn.Module,
    args,
    inputs: torch.Tensor,
    pruning_ratio: float,
) -> tuple:
    """Apply global structured channel pruning using magnitude importance.

    .. warning::
        This function is **not idempotent**. Every call permanently modifies
        the given *model* instance in-place. Reload the checkpoint before
        calling again if you need a clean model.

    The final output layer (``nn.Linear`` whose ``out_features`` equals
    ``args.window_size``) is automatically added to ``ignored_layers`` so
    it is never pruned.  For TCN models (which have no ``nn.Linear``
    layers) this list remains empty.

    Parameters
    ----------
    model         : nn.Module       — the model to prune (modified in-place)
    args          : SimpleNamespace — must expose ``args.window_size`` (int):
                    the ``out_features`` of the output Linear to protect;
                    set to ``1`` for CNN (Seq2Point) or the sequence length
                    for TCN (though TCN has no Linear so it is irrelevant)
    inputs        : torch.Tensor    — a dummy input for dependency-graph tracing
    pruning_ratio : float           — fraction of channels to remove (default 0.5)

    Returns
    -------
    model  : nn.Module     — the pruned model (same object, modified in-place)
    output : torch.Tensor  — forward-pass output used to verify correctness
    """
    import torch_pruning as tp

    # GRU models cannot be structured-pruned with MetaPruner: bidirectional GRU
    # weight pairs (weight_ih_l0 / weight_ih_l0_reverse, etc.) must stay in sync
    # but MetaPruner has no built-in handler for them and prunes each independently,
    # causing a shape mismatch when PyTorch calls flatten_parameters() internally.
    # Ignoring the GRU layers also does not help because conv1→gru1 and gru2→fc1
    # dependency edges freeze the entire channel graph.
    for m in model.modules():
        if isinstance(m, (torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN)):
            raise NotImplementedError(
                "Structured channel pruning with torch_pruning is not supported for "
                "models containing GRU/LSTM/RNN layers. "
                "Consider weight pruning (unstructured) or knowledge distillation instead."
            )

    imp = tp.importance.MagnitudeImportance(p=1)

    pruning_channel = param_ratio_to_channel_ratio(pruning_ratio)

    # Protect the final output layer from pruning
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            if m.out_features == args.window_size:
                ignored_layers.append(m)

    pruner = tp.pruner.MetaPruner(
        model,
        inputs,
        importance=imp,
        pruning_ratio=pruning_channel,
        global_pruning=True,
        isomorphic=True,
        iterative_steps=1,
        ignored_layers=ignored_layers,
    )

    base_macs, base_params = count_ops_and_params(model, inputs)
    pruner.step()
    pruned_macs, pruned_params = count_ops_and_params(model, inputs)

    print(f"Baseline model  — MACs: {base_macs:,}  |  Params: {base_params:,}")
    print(f"Pruned model    — MACs: {pruned_macs:,}  |  Params: {pruned_params:,}")
    print(f"MACs reduction  : {(1 - pruned_macs / base_macs) * 100:.1f}%")
    print(f"Param reduction : {(1 - pruned_params / base_params) * 100:.1f}%")

    output = model(inputs)
    print(f"Output shape    : {output.shape}")

    return model, output


# =============================================================================
# 3. Unstructured Pruning
# =============================================================================

def get_prunable_parameters(model: nn.Module) -> list:
    """Return a list of ``(module, name)`` tuples for global unstructured pruning.

    Covers all weight tensors in:
    - ``nn.Conv1d`` / ``nn.Conv2d`` / ``nn.Linear``   → ``'weight'``
    - ``nn.GRU``                                        → all ``weight_*`` parameters
      (``weight_ih_lN``, ``weight_hh_lN``, and bidirectional variants)

    Bias tensors are intentionally excluded — pruning biases offers negligible
    compression and can destabilise training.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    list of (module, param_name) tuples — ready to pass to
    ``torch.nn.utils.prune.global_unstructured``
    """
    params = []
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            params.append((module, 'weight'))
        elif isinstance(module, nn.GRU):
            for name, _ in module.named_parameters(recurse=False):
                if 'weight' in name:
                    params.append((module, name))
    return params


def apply_unstructured_pruning(model: nn.Module, amount: float) -> nn.Module:
    """Apply global L1 unstructured pruning to *model*.

    Uses :func:`torch.nn.utils.prune.global_unstructured` with
    ``L1Unstructured`` importance, which zeros the *amount* fraction of weights
    with the smallest absolute value across all prunable layers simultaneously.

    The model is modified **in place** by attaching pruning forward-pre-hooks.
    The original weights are stored in ``<name>_orig`` buffers; the binary mask
    is stored in ``<name>_mask`` buffers.  Both are removed when you call
    :func:`remove_pruning_masks`.

    Works with **CNN, GRU, and TCN** (unlike structured pruning which cannot
    handle GRU/LSTM layers).

    Parameters
    ----------
    model  : nn.Module
    amount : float — fraction of weights to zero, in ``(0, 1)``

    Returns
    -------
    nn.Module — the same *model* object with pruning hooks attached

    Raises
    ------
    ValueError — if no prunable parameters are found
    """
    import torch.nn.utils.prune as prune

    if not 0.0 < amount < 1.0:
        raise ValueError(f"amount must be in (0, 1), got {amount}")

    parameters = get_prunable_parameters(model)
    if not parameters:
        raise ValueError("No prunable parameters found in the model.")

    prune.global_unstructured(
        parameters,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # Use getattr on each (module, name) pair — for GRU the hook redirects
    # name → name_orig * name_mask, so this correctly reflects the masked weight.
    total = zeros = 0
    for m, n in parameters:
        w      = getattr(m, n)
        total += w.numel()
        zeros += int((w == 0).sum())
    actual_sparsity = zeros / total if total > 0 else 0.0
    print(f"Unstructured pruning applied — amount: {amount:.0%}")
    print(f"Global sparsity : {actual_sparsity * 100:.1f}%  ({zeros:,} / {total:,} weights zeroed)")

    return model


def remove_pruning_masks(model: nn.Module) -> nn.Module:
    """Make unstructured pruning permanent by baking masks into the weights.

    Removes all pruning forward-pre-hooks and the ``_mask`` / ``_orig`` buffers
    from every module, leaving a clean model whose ``state_dict`` contains only
    the standard ``weight`` tensors (with zeroed entries for pruned weights).

    Should be called **after** fine-tuning and **before** saving the checkpoint.

    Parameters
    ----------
    model : nn.Module — model with active pruning hooks

    Returns
    -------
    nn.Module — the same object, hooks removed, weights baked in
    """
    import torch.nn.utils.prune as prune

    for module in model.modules():
        # After pruning, each pruned parameter 'name' is replaced by 'name_orig'
        # in the module's _parameters dict.  Strip the '_orig' suffix to recover
        # the name expected by prune.remove().
        orig_names = [
            n for n in list(module._parameters.keys())
            if n.endswith('_orig')
        ]
        for orig_name in orig_names:
            param_name = orig_name[:-5]  # strip '_orig'
            prune.remove(module, param_name)

    return model


def get_model_sparsity(model: nn.Module) -> float:
    """Return the global weight sparsity (fraction of zero weights) of *model*.

    Scans Conv1d, Conv2d, Linear, and GRU weight tensors.
    Works both before and after :func:`remove_pruning_masks` — when pruning hooks
    are active, the *effective* weight (``weight_orig * weight_mask``) is used.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    float — fraction of zero weights in ``[0, 1]``
    """
    total = zeros = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            w      = module.weight
            total += w.numel()
            zeros += int((w == 0).sum())
        elif isinstance(module, nn.GRU):
            # When pruning masks are active, _parameters holds 'weight_ih_l0_orig'
            # instead of 'weight_ih_l0'.  We strip '_orig' to get the base name,
            # then use getattr() which goes through the hook and returns the
            # effective masked weight (orig * mask).
            weight_base_names: set = set()
            for pname in list(module._parameters.keys()):
                base = pname[:-5] if pname.endswith('_orig') else pname
                if 'weight' in base:
                    weight_base_names.add(base)
            for wname in weight_base_names:
                w      = getattr(module, wname)
                total += w.numel()
                zeros += int((w == 0).sum())
    return zeros / total if total > 0 else 0.0
