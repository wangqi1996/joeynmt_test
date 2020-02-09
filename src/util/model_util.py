# coding=utf-8

from torch import nn


def freeze_params(model: nn.Module) -> None:
    """
    Freeze the parameters of this model
    i.e. do not update them during training

    """

    for _, p in model.named_parameters():
        p.requires_grad = False

    # for p in model.parameters():
    #     p.requires_grad = False
