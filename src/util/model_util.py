# coding=utf-8
import random

import numpy as np
import torch
from torch import nn, Tensor


def freeze_params(model: nn.Module) -> None:
    """
    Freeze the parameters of this module
    i.e. do not update them during training

    """

    for _, p in model.named_parameters():
        p.requires_grad = False

    # for p in module.parameters():
    #     p.requires_grad = False


def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


def set_seed(seed: int) -> None:
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
