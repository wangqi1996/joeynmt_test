# coding=utf-8
"""
Module to represents whole models
"""

from torch import nn, Tensor

from src.model.decoder import Decoder
from src.model.encoder import Encoder
from src.module.embedding import Embeddings
from src.util.constants import BOS_TOKEN, PAD_TOKEN, EOS_TOKEN
from src.util.vocabulary import Vocabulary


class Model(nn.Module):
    """
    Base model class
    """

    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed: Embeddings, trg_embed: Embeddings,
                 src_vocab: Vocabulary, trg_vocab: Vocabulary):
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param src_embed: source embedding
        :param trg_embed: target embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        """
        super(Model, self).__init__()

        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.encoder = encoder
        self.decoder = decoder
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        self.bos_index = self.trg_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.trg_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.trg_vocab.stoi[EOS_TOKEN]

    def forward(self, src: Tensor, trg_input: Tensor, src_mask: Tensor,
                src_lengths: Tensor, trg_mask: Tensor = None) -> (
            Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param src: source input
        :param trg_input: target input
        :param src_mask: source mask
        :param src_lengths: length of source inputs
        :param trg_mask: target mask
        :return: decoder outputs
        """
        encoder_output, encoder_hidden = self.encode(src=src,
                                                     src_length=src_lengths,
                                                     src_mask=src_mask)
        unroll_steps = trg_input.size(1)
        return self.decode(encoder_output=encoder_output,
                           encoder_hidden=encoder_hidden,
                           src_mask=src_mask, trg_input=trg_input,
                           unroll_steps=unroll_steps,
                           trg_mask=trg_mask)

    def encode(self, src: Tensor, src_length: Tensor, src_mask: Tensor) -> (Tensor, Tensor):
        """
        Encodes the source sentence.
        """
        src_emb = self.src_embed(src)
        return self.encoder(src_emb, src_length, src_mask)

    def decode(self, encoder_output: Tensor, encoder_hidden: Tensor,
               src_mask: Tensor, trg_input: Tensor,
               unroll_steps: int, decoder_hidden: Tensor = None,
               trg_mask: Tensor = None) \
            -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        tgt_emb = self.trg_embed(trg_input)
        return self.decoder(trg_embed=tgt_emb,
                            encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=src_mask,
                            unroll_steps=unroll_steps,
                            hidden=decoder_hidden,
                            trg_mask=trg_mask)
