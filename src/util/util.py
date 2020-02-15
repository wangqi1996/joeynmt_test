# coding=utf-8
import numpy as np
import torch
import yaml
from torch import Tensor


class ConfigurationError(Exception):
    """ Custom exception for misspecifications of configuration """
    pass




def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.

    不是repeat！！！！
    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """

    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    # 排列, 重复的维度是第0维度
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()

    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)

    x = x.view(batch, -1) \
        .transpose(0, 1) \
        .repeat(count, 1) \
        .transpose(0, 1) \
        .contiguous() \
        .view(*out_size)

    if dim != 0:
        x = x.permute(perm).contiguous()

    return x


def load_config(path="config/default.ymal") -> dict:
    """
    Load and parses a YAML configuration file

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg
