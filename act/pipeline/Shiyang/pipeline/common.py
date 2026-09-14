from __future__ import annotations

import torch
import torch.nn as nn

from act.front_end.spec_creator_base import LabeledInputTensor
from act.front_end.verifiable_model import InputLayer, OutputSpecLayer


def initial_seeds_from_wrapped_model(wrapped_model: nn.Module) -> list[LabeledInputTensor]:
    """Extract per-instance seeds from an already-batched wrapped model.

    VNNLIB-sourced instances never populate ``InputLayer.labeled_input.label``
    -- the true class instead lives on the OutputSpecLayer's own spec
    (``OutputSpec.y_true``), derived from the property's constraints (see
    act/pipeline/fuzzing/pattern_search_pgd.py's identical workaround, same
    root cause). Falling back to label=None (=> label=-1 in FuzzingSeed)
    would make PGDMutation/HPGDMutation silently drop to their unlabeled
    "maximize output variance" objective instead of the targeted CW-margin
    loss toward y_true, and would make PropertyChecker unable to tell which
    class is "correct" -- so we backfill from OutputSpec.y_true whenever a
    per-instance label wasn't already supplied.
    """
    input_layer = next(
        (module for module in wrapped_model.modules() if isinstance(module, InputLayer)),
        None,
    )
    if input_layer is None:
        raise AttributeError("wrapped model does not contain an InputLayer")

    output_spec_layer = next(
        (module for module in wrapped_model.modules() if isinstance(module, OutputSpecLayer)),
        None,
    )
    y_true = getattr(getattr(output_spec_layer, "spec", None), "y_true", None)

    labeled_input = input_layer.labeled_input
    tensors = labeled_input.tensor
    labels = labeled_input.label
    seeds = []
    for i in range(tensors.shape[0]):
        label = labels[i : i + 1] if labels is not None and labels.numel() > 0 else None
        if label is None and y_true is not None:
            y_true_t = y_true if isinstance(y_true, torch.Tensor) else torch.tensor(y_true)
            # y_true is [B] (one true class per batched instance) or [1]
            # (shared across all instances, e.g. a single-instance batch).
            label = y_true_t[i : i + 1] if y_true_t.numel() > 1 else y_true_t.view(-1)[:1]
        seeds.append(LabeledInputTensor(tensor=tensors[i : i + 1], label=label))
    return seeds
