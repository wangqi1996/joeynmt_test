# coding=utf-8
"""
Module to represents whole models
"""

import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor

from src.data.batch import Batch
from src.data.vocabulary import Vocabulary
from src.model.decoder import Decoder
from src.model.encoder import Encoder
from src.module.embedding import Embeddings
from src.util.constants import BOS_TOKEN, PAD_TOKEN, EOS_TOKEN


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

    def get_loss_for_batch(self, batch: Batch, loss_function: nn.Module) -> Tensor:
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param loss_function:  loss function, computes for input and target
        a scalar loss for the complete batch

        :return: batch_loss: sum of losses over non-pad elements in the batch
        """
        out, hidden, att_probs, _ = self.forward(
            src=batch.src, trg_input=batch.trg_input,
            src_mask=batch.src_mask, src_lengths=batch.src_length,
            trg_mask=batch.trg_mask
        )

        # compute log probs
        log_probs = F.log_softmax(out, dim=1)

        # compute batch loss
        batch_loss = loss_function(log_probs, batch.trg)

        # return batch_loss = sum over all elements in batch that not pad
        return batch_loss

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings
        :return: string representation
        """
        return "%s(\n" \
               "\tencoder=%s,\n" \
               "\tdecoder=%s,\n" \
               "\tsrc_embed=%s,\n" \
               "\ttrg_embed=%s)" % (self.__class__.__name__, self.encoder,
                                    self.decoder, self.src_embed, self.trg_embed)

    def run_batch(self, batch: Batch, max_output_length: int, beam_size: int,
                  beam_alpha: float) -> (np.array, np.array):
        """
        get outputs and attentions scores for a given batch

        :param batch:  batch to generate hypotheses for
        :param max_output_length:  maximum length of hypotheses
        :param beam_size:  size of the beam for beam search, if 0 use greedy
        :param beam_alpha:  hypotheses for batch
        :return:  stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """
        encoder_output, encoder_hidden = self.encode(
            batch.src, batch.src_length, batch.src_mask
        )

        #  if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(max(batch.src_length.cpu().numpy()) * 1.5)

        # greedy decoding
        if beam_size < 2:
            stacked_output, stacked_attention_scores = greedy(
                encoder_hidden=encoder_hidden,
                encoder_output=encoder_output, eos_index=self.eos_index,
                src_mask=batch.src_mask, embed=self.trg_embed,
                bos_index=self.bos_index, decoder=self.decoder,
                max_output_length=max_output_length
            )
        else:
            stacked_output, stacked_attention_scores = \
                beam_search(
                    size=beam_size, encoder_output=encoder_output,
                    encoder_hidden=encoder_hidden,
                    src_mask=batch.src_mask, embed=self.trg_embed,
                    max_output_length=max_output_length,
                    alpha=beam_alpha, eos_index=self.eos_index,
                    pad_index=self.pad_index,
                    bos_index=self.bos_index,
                    decoder=self.decoder)

        return stacked_output, stacked_attention_scores
