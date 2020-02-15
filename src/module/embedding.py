# coding=utf-8
import math
import torch
from torch import Tensor
from torch import nn

from src.util.model_util import freeze_params


class Embeddings(nn.Module):
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
            # transformer
            return self.embedding(x) * math.sqrt(self.embedding_dim)

        return self.embedding(x)

    def __repr__(self):
        return "%s(embedding_dim=%d, vocab_size=%d)" % (
            self.__class__.__name__, self.embedding_dim, self.vocab_size)


class PositionalEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    In forward pass, this adds the position-encodings to the
    input for as many time steps as necessary.

    Implementation based on OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, size: int = 0, max_len: int = 5000):
        """

        Positional Encoding with maximum length max_len

        :param size:  module dim
        :param max_len:
        """

        if size % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(size))

        pe = torch.zeros(max_len, size)
        position = torch.arange(1, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, size, 1, dtype=torch.float)
                              * - (math.log(10000.0) / size)
                              ))
        # 两个::表示步长
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        pe = pe.unsqueeze(0)  # shape [1, max_len, size]
        super().__init__()

        # Buffers won’t be returned in module.parameters(), so that the optimizer won’t have a change to update them.
        # buf can return in module.OrderDict(), so can be save with the module.save()
        self.register_buffer('pe', pe)
        self.dim = size

    def forward(self, emb):
        """
        Embed inputs

        :param emb (FloatTensor): Sequence of word vectors
        :return: (batch_size, seq_len, self.dim)
        """
        # Add position encoding
        return emb + self.pe[:, :emb.size(1)]
