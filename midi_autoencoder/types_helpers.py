"""Module to enforce coherent type hints in the library.

Contains few TypedDict classes:
- OutputEncoder
- OutputModel
- OutputLoss

"""

from torch import Tensor
from typing import TypedDict


class EncoderOutput(TypedDict):
    """TypedDict for normalizing outputs of all Encoder."""

    mu: Tensor
    log_var: Tensor
    pre_latents: Tensor


class ModelOutput(TypedDict):
    """TypedDict for normalizing outputs of all VAE models."""

    output: Tensor
    input: Tensor
    encoded: EncoderOutput
    latents: Tensor


class LossOutput(TypedDict):
    """TypedDict for normalizing outputs of all losses."""

    loss: Tensor
    reconstruction_loss: Tensor
    kld_loss: Tensor
