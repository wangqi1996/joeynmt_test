# coding=utf-8
import math
from torch import Tensor
from torch import nn

from src.util.model_util import freeze_params


class Embedding(nn.Module):
    """

    simple embedding class
    """

    def __init__(self, embedding_dim: int = 64, scale: bool = False, vocab_size: int = 0,
                 padding_idx: int = 1, freeze: bool = False, **kwargs):
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.

        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.scale = scale

        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=padding_idx)

        if freeze:
            freeze_params(self)

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform lookup for input `x` in the embedding table.

        """
        if self.scale:
            return self.embedding(x) * math.sqrt(self.embedding_dim)

        return self.embedding(x)

    def __repr__(self):
        return "%s(embedding_dim=%d, vocab_size=%d)" % (
            self.__class__.__name__, self.embedding_dim, self.vocab_size)
