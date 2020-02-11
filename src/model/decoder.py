# coding=utf-8

import torch
from torch import nn

class Decoder(nn.Module):
    """
    Base decoder class
    """

    @property
    def output_size(self):
        """
        Return the output size (size of the target vocabulary)
        :return:
        """
        return self._output_size



