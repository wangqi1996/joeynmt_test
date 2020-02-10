# coding=utf-8

"""
Attention modules
"""
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor


class AttentionMechanism(nn.Module):
    """
    Base attention class
    """

    def __init__(self):
        pass

    def forward(self, *inputs):
        raise NotImplementedError("AttentionMechanism is base class!!")


class BahdanauAttention(AttentionMechanism):
    """
    Implements Bahdanau (MLP) attention

    Section A.1.2 in https://arxiv.org/pdf/1409.0473.pdf.
    """

    def __init__(self, hidden_size=1, key_size=1, query_size=1):
        """
        Creates attention mechanism.

        TODO: hidden_size = query_size
        TODO: key_size=value_size
        :param hidden_size: size of the projection for query and key
        :param key_size: size of the attention input keys
        :param query_size: size of the query
        """
        super().__init__()

        assert query_size == hidden_size, u"原来真的有query_size != hidden_size的情况"
        self.key_layer = nn.Linear(in_features=key_size, out_features=hidden_size, bias=False)
        self.query_layer = nn.Linear(in_features=query_size, out_features=hidden_size, bias=False)
        self.energy_layer = nn.Linear(in_features=hidden_size, out_features=1, bias=False)

        self.proj_keys = None  # to store projected keys
        self.proj_query = None  # to store projected query

    def forward(self, query: Tensor = None, mask: Tensor = None, values: Tensor = None):
        """
        Bahdanau MLP attention forward pass

        :param query: the item(decoder state) to compare with the keys/memory
                shape (batch_size, 1, decoder.hidden_size)
        :param mask: mask out keys position (0 is invalid positions, 1 else)
                shape (batch_size, 1, decoder.hidden_size)
        :param values: valus (encoder states)
                shape (batch_size, src_length, encoder.hidden_size)
        :return: contxt vector of shape (batch_size, 1, src_length)
                attention probabilities of shape (batch_size, 1, src_length)
        """
        # check
        assert mask is not None, u"mask in not None"
        assert self.proj_keys is not None, \
            "projection keys have to get pre-computed"

        self._check_input_shapes_forward(query=query, mask=mask, values=values)

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        self.compute_proj_query(query)

        # calculate scores
        # proj_keys: batch, src_len, hidden_size
        # proj_query: batch, 1, hidden_size
        # scores: batch, src_len, 1
        scores = self.energy_layer(torch.tanh(self.proj_keys + self.proj_query))

        # scores: batch, 1, src_len
        scores = scores.squeeze(2).unsqueeze(1)

        # mask (0 is invalid positions, 1 else)  -> (1 is invalid positions, 0 else)
        mask = mask == 0
        scores = scores.masked_fill(mask, float('-inf'))

        # turn scores to probabilities: batch_size, 1, src_len
        alphas = F.softmax(scores, dim=-1)

        # compute the context
        context = alphas @ values  # matmul

        return context, alphas

    def compute_proj_keys(self, keys: Tensor):
        """
        Compute the projection of the keys.
        Is efficient if pre-computed before receiving individual queries.
        :param keys:
        :return:
        """
        self.proj_keys = self.key_layer(keys)

    def compute_proj_query(self, query: Tensor):
        """
        Compute the projection of the query.
        :param query:
        :return:
        """
        self.proj_query = self.query_layer(query)

    def _check_input_shapes_forward(self, query: Tensor, mask: Tensor, values: Tensor):
        """
        Make sure that inputs to 'self.forward' are of correct shape

        Same input semantics as for 'self.forward'
        """

        # batch_size
        assert query.shape[0] == values.shape[0] == mask.shape[0]

        # 维度1
        assert query.shape[1] == 1 == mask.shape[1]

        # query_size = decoder.hidden_size
        assert query.shape[2] == self.query_layer.in_features

        # src_length
        assert mask.shape[2] == values.shape[1]

        # value_size == key_size
        assert values.shape[2] == self.key_layer.in_features

    def __repr__(self):
        return 'BahdanauAttention'


class LuongAttention(AttentionMechanism):
    """
    Implements Luong (bilinear / multiplicative) attention.

    Eq. 8 ("general") in http://aclweb.org/anthology/D15-1166.

    """

    def __init__(self, hidden_size: int = 1, key_size: int = 1):
        """
        Creates attention mechanism.

        :param hidden_size: size of the key projection layer, has to be equal
            to decoder hidden size
        :param key_size: size of the attention input keys
        """
        super().__init__()

        self.key_layer = nn.Linear(in_features=key_size, out_features=hidden_size, bias=False)

        self.proj_keys = None  # projected keys

    def forward(self, query: Tensor = None, mask: Tensor = None, values: Tensor = None):
        """
        Luong (multiplicative / bilinear) attention forward pass.

        Computes context vectors and attention scores for a given query and
        all masked values and returns them.

        :param query: the item (decoder state) to compare with the keys/memory,
            shape (batch_size, 1, decoder.hidden_size)
        :param mask: mask out keys position (0 in invalid positions, 1 else),
            shape (batch_size, 1, src_length)
        :param values: values (encoder states),
            shape (batch_size, src_length, encoder.hidden_size)
        :return: context vector of shape (batch_size, 1, value_size),
            attention probabilities of shape (batch_size, 1, src_length)
        """
        self._check_input_shapes_forward(query=query, mask=mask, values=values)
        assert self.proj_keys is not None, u"projections keys have to get pre-compute"
        assert mask is not None, "mask is required"

        scores = query @ self.proj_keys.transpose(1, 2)

        # mask
        mask = mask == 0
        scores = scores.masked_fill(mask, float('-inf'))

        # turn scores to probabilities
        alphas = F.softmax(scores, dim=-1)

        context = alphas @ values

        return context, alphas

    def _check_input_shapes_forward(self, query: Tensor = None, mask: Tensor = None, values: Tensor = None):
        """
        Make sure that inputs to `self.forward` are of correct shape.
        Same input semantics as for `self.forward`.
        """
        # batch_size
        assert query.shape[0] == values.shape[0] == mask.shape[0]

        # w dim=1
        assert query.shape[1] == 1 == mask.shape[1]

        # hidden_dim
        assert query.shape[2] == self.key_layer.out_features

        # value_dim = key_dim
        assert values.shape[2] == self.key_layer.in_features

        # src_len
        assert mask.shape[2] == values.shape[1]

    def compute_proj_keys(self, keys: Tensor):
        """
        Compute the projection of the keys and assign them to `self.proj_keys`.

        This pre-computation is efficiently done for all keys before receiving individual queries.

        :param keys: shape (batch_size, src_length, encoder.hidden_size)
        """
        # proj_keys: batch x src_len x hidden_size
        self.proj_keys = self.key_layer(keys)

    def __repr__(self):
        return "LuongAttention"


class MultiHeadedAttention(AttentionMechanism):
    """
    Multi-Head Attention module from "Attention is All You Need"

    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, num_heads: int, size: int, dropout: float = 0.1):
        """
        create a multi-headed attention layer

        :param num_heads:
        :param size: module size (must be divisible by num_heads)
        :param dropout:
        """
        super().__init__()

        assert size % num_heads == 0

        self.head_size = size // num_heads
        self.model_size = size
        self.dropout = dropout
        self.num_heads = num_heads

        # 多个层的映射层拼接在一起
        self.k_layer = nn.Linear(size, num_heads * self.head_size)
        self.v_layer = nn.Linear(size, num_heads * self.head_size)
        self.q_layer = nn.Linear(size, num_heads * self.head_size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):
        """
        Compute multi-headed attention

        :param k:  keys [B, M, D] with M being the sentence length
        :param v:  values [B, M, D]
        :param q:  query [B, M, D], v=q
        :param mask:  optional mask [B, 1, M]
        :return:
        """
        batch_size = k.size(0)
        num_heads = self.num_heads

        # project the queries (q), keys(k). values(v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q,k,v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)

        # compute scores
        q = q / math.sqrt(self.head_size)

        # scores = q @ k.transpose(2, 3)
        # [b, num_heads, q_len, head_size] * [b, num_heads, head_size, k_len]
        #  = [b, num_heads, q_len, k_len]
        scores = torch.matmul(q, k.transpose(2, 3))

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))

        alpha = F.softmax(scores)

        # apply dropout
        alpha = self.dropout(alpha)

        # [b, num_size, q_len, head_size]  q_len = v_len
        attention = alpha @ v

        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, num_heads * self.head_size)
        output = self.output_layer(attention)

        return output
