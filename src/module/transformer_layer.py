# coding=utf-8
from torch import Tensor
from torch import nn

from src.module.attention import MultiHeadedAttention


class PositionwiseFeedForward(nn.Module):
    """

    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.

    """

    def __init__(self, input_size, ff_size, dropout=0.1):
        """
        Initializes position-wise feed-forward layer

        :param input_size:  dimensionality of the input.
        :param ff_size:  dimensionality of intermediate representation
        :param dropout:
        """

        super().__init__()
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
        )

    def forward(self, x):
        return self.pwff_layer(x)

class TransformerEncoderLayer(nn.Module):
    """
    One Transformer encoder layer has a Multi-head attention layer plus a position-wise feed-forward layer.
    """

    def __init__(self, size: int = 0, ff_size: int = 0,
                 num_heads: int = 0, dropout: float = 0.1):
        """
        A single Transformer layer.
        """
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(size, eps=1e-6)
        self.src_src_att = MultiHeadedAttention(num_heads=num_heads, size=size, dropout=dropout)
        self.layer_norm2 = nn.LayerNorm(size, eps=1e-6)
        self.feed_forward = PositionwiseFeedForward(input_size=size, ff_size=ff_size, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.size = size

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass for a single transformer encoder layer.
        First applies layer norm, then self attention,
        then dropout with residual connection (adding the input to the result),
        and then a position-wise feed-forward layer.

        :param x:
        :param mask:
        :return:
        """
        # layer_norm + attention
        x_norm = self.layer_norm1(x)
        h = self.src_src_att(x_norm, x_norm, x_norm, mask)
        attention = self.dropout(h) + x
        # layer_norm + FFN
        attention_norm = self.layer_norm2(attention)
        h = self.feed_forward(attention_norm)
        output = self.dropout(h) + attention

        return output


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer.
    Consists of self-attention, source-attention, and feed-forward.
    """

    def __init__(self, size: int = 0, ff_size: int = 0,
                 num_heads: int = 0, dropout: float = 0.1):
        """
        Represents a single Transformer decoder layer.
        It attends to the source representation and the previous decoder states.

        :param size: module dimensionality
        :param ff_size: size of the feed-forward intermediate layer
        :param num_heads: number of heads
        :param dropout: dropout to apply to input
        """

        super().__init__()
        self.size = size

        self.layer_norm1 = nn.LayerNorm(size, eps=1e-6)
        self.tgt_tgt_att = MultiHeadedAttention(num_heads, size, dropout=dropout)

        self.layer_norm2 = nn.LayerNorm(size, eps=1e-6)
        self.src_tgt_att = MultiHeadedAttention(num_heads, size, dropout=dropout)

        self.layer_norm3 = nn.LayerNorm(size, eps=1e-6)
        self.feed_forward = PositionwiseFeedForward(input_size=size, ff_size=ff_size, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor = None, memory: Tensor = None,
                src_mask: Tensor = None, tgt_mask: Tensor = None, ):
        """
        Forward pass of a single Transformer decoder layer.

        :param x: inputs
        :param memory: source representations
        :param src_mask:  source mask
        :param tgt_mask:  target mask (so as to not condition on future steps)
        :return: output tensor
        """
        x_norm = self.layer_norm1(x)
        attention = self.tgt_tgt_att(x_norm, x_norm, x_norm, tgt_mask)
        intra_attention = self.dropout(attention) + x

        attention_norm = self.layer_norm2(intra_attention)
        attention = self.src_tgt_att(k=memory, v=memory, q=attention_norm, mask=src_mask)
        inter_attention = self.dropout(attention) + intra_attention

        attention_norm = self.layer_norm3(inter_attention)
        FFN_output = self.feed_forward(attention_norm)
        FFN_output = self.dropout(FFN_output) + inter_attention

        return FFN_output
