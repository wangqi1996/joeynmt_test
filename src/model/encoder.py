# coding=utf-8
import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.module.embedding import PositionalEncoding
from src.module.transformer_layer import TransformerEncoderLayer
from src.util.model_util import freeze_params


class Encoder(nn.Module):
    """
    base encoder class
    """

    @property
    def output_size(self):
        """
        return the output size
        :return:
        """
        return self._output_size


class RecurrentEncoder(Encoder):
    """
    Encoders a sequence of word embedding
    """

    def __init__(self, rnn_type: str = 'gru', hidden_size: int = 1, emb_size: int = 1,
                 num_layers: int = 1, dropout: float = 0., emb_dropout: float = 0.,
                 bidirectional: bool = True, freeze: bool = False, **kwargs) -> None:
        """
        Create a new recurrent encoder.

        :param rnn_type: RNN type: `gru` or `lstm`.
        :param hidden_size: Size of each RNN.
        :param emb_size: Size of the word embeddings.
        :param num_layers: Number of encoder RNN layers.
        :param dropout:  Is applied between RNN layers.
        :param emb_dropout: Is applied to the RNN input (word embeddings).
        :param bidirectional: Use a bi-directional RNN.
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super().__init__()
        self.emb_dropout = nn.Dropout(p=emb_dropout, inplace=False)
        self.type = type
        self.emb_size = emb_size

        rnn = nn.GRU if rnn_type == 'gru' else nn.LSTM

        dropout = dropout if num_layers > 1 else 0.
        self.rnn = rnn(
            emb_size, hidden_size, num_layers, batch_first=True,
            bidirectional=bidirectional, dropout=dropout
        )

        self._output_size = 2 * hidden_size if bidirectional else hidden_size

        if freeze:
            freeze_params(self)

    def _check_shape_input_forward(self, embed_src: Tensor, src_length: Tensor,
                                   mask: Tensor) -> None:
        """
         Make sure the shape of the inputs to `self.forward` are correct.
         Same input semantics as `self.forward`.

         :param embed_src: embedded source tokens
         :param src_length: source length
         :param mask: source mask
         """

        assert embed_src.shape[2] == self.emb_size
        assert embed_src.shape[0] == src_length.shape[0]
        assert src_length.shape[1] == 1

    def forward(self, embed_src: Tensor, src_length: Tensor, mask: Tensor):
        """
         Applies a bidirectional RNN to sequence of embeddings x.
         The input mini-batch x needs to be sorted by src length.
         x and mask should have the same dimensions [batch, time, dim].

         :param embed_src: embedded src inputs,
             shape (batch_size, src_len, embed_size)
         :param src_length: length of src inputs
             (counting tokens before padding), shape (batch_size)
         :param mask: indicates padding areas (zeros where padding), shape
             (batch_size, src_len, embed_size)

         :return:
             - output: hidden states with
                 shape (batch_size, max_length, directions*hidden),
             - hidden_concat: last hidden state with
                 shape (batch_size, directions*hidden)
         """
        self._check_shape_input_forward(embed_src, src_length, mask)

        # apply dropout to emb_src
        embed_src = self.emb_dropout(embed_src)

        packed = pack_padded_sequence(embed_src, src_length, batch_first=True)
        output, hidden = self.rnn(packed)

        # lstm输出的是一个tuple
        if isinstance(hidden, tuple):
            hidden, memory_cell = hidden

        output, _ = pad_packed_sequence(output, batch_first=True)

        # hidden: bidirectional * layers, batch, hidden  最后一个时间步
        # output: batch, max_length, bidirectional * hidden  # 最后一层的
        batch_size = hidden.size()[1]

        # separate final hidden states by layer and direction
        # [layers, bidirectional, batch, hidden_size]
        hidden_layerwise = hidden.view(self.rnn.num_layers, 2 if self.rnn.bidirectional else 1,
                                       batch_size, self.rnn.hidden_size)

        # concat the final states of the last layer for each directions
        # thanks to pach_padded_sequence final states do not include padding
        # [1, batch_size, hidden_size]
        fwd_hidden_last = hidden_layerwise[-1:, 0]
        bwd_hidden_last = hidden_layerwise[-1:, 1]
        """
        fwd_hidden_last 作用等同于：hidden[-2, :], output[:, -1, :hidden_size] (正向)
        bwd_hidden_lsat 作用等同于：hidden[-1, :], output[:, 0, hidden_size:]
        """

        # only feed the final state of the top-most layer to the decoder
        # [batch_size, hidden_size * bidrections]
        hidden_concat = torch.cat(
            [fwd_hidden_last, bwd_hidden_last], dim=2).squeeze(0)

        # final batch, bidirections * hidden
        return output, hidden_concat

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.rnn)


class TransformerEncoder(Encoder):
    """
    Transformer Encoder
    """

    def __init__(self, hidden_size: int = 512, ff_size: int = 2048, num_layers: int = 8,
                 num_heads: int = 4, dropout: float = 0.1, emb_dropout: float = 0.1,
                 freeze: bool = False, **kwargs):
        """
        Initializes the Transformer.

        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super().__init__()

        # build all (num_kayers) layers
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(size=hidden_size, ff_size=ff_size,
                                     num_heads=num_heads, dropout=dropout)
             for _ in range(num_layers)]
        )

        # add the top transformer ffn layer
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self._output_size = hidden_size

        if freeze:
            freeze_params(self)

    def forward(self, embed_src: Tensor, src_length: Tensor, mask: Tensor) -> (Tensor, Tensor):
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """

        x = self.pe(embed_src)
        x = self.emb_dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        x_norm = self.layer_norm(x)
        return x_norm, None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].src_src_att.num_heads)
