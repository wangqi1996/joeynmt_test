# coding=utf-8

from torch import nn


def freeze_params(model: nn.Module) -> None:
    """
    Freeze the parameters of this module
    i.e. do not update them during training

    """

    for _, p in model.named_parameters():
        p.requires_grad = False

    # for p in module.parameters():
    #     p.requires_grad = False
