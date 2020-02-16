# coding=utf-8
import errno
import logging
import os
import shutil

import torch
import yaml
from torch import Tensor


class ConfigurationError(Exception):
    """ Custom exception for misspecifications of configuration """
    pass


def make_model_dir(model_dir: str, overwrite=False) -> str:
    """
    Create a new dictionary for the model

    :param model_dir:  path to model dictionary
    :param overwrite:  whether to overwrite an existing directory
    :return:  path to model directory
    """
    if os.path.isdir(model_dir):
        if not overwrite:
            raise FileExistsError(
                "Model directory exists and overwriting is disabled.")

        # delete previous directory to start with empty dir again
        shutil.rmtree(model_dir)

    os.makedirs(model_dir)
    return model_dir


def make_logger(log_file: str = None) -> logging.Logger:
    """
    Create a logger for logging the training/testing process

    :param log_file: path to file where log is stored as well
    :return:  logger object
    """

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(message)s")

    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level=logging.DEBUG)
        logger.addHandler(fh)
        fh.setFormatter(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logging.getLogger("").addHandler(sh)
    logging.info("hello! This is Joey-NMT")

    return logger


def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    """
    Load model form saved checkpoint

    :param path: path to checkpoint
    :param use_cuda:  using cuda or not
    :return:  checkpoint(dict)
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location="cuda" if use_cuda else "cpu")
    return checkpoint


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


def symlink_update(target, link_name):
    try:
        os.symlink(target, link_name)
    except FileExistsError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e
