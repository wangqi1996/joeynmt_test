# coding=utf-8

"""
Module to implement training loss
"""
import torch
from torch import nn, Tensor
from torch.autograd import Variable


class XentLoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing
    """

    def __int__(self, pad_index: int, smoothing: float = 0.0):
        super().__init__()

        self.smoothing = smoothing
        self.pad_index = pad_index

        """
        KLDivLoss要求target和input的维度相同
        """
        if self.smoothing <= 0.0:
            # CrossEntropyLoss其实是LogSoftMax和NLLLoss的合体，也就是所有的loss都会先经历一次log SoftMax之后再进入交叉熵公式。
            self.critertion = nn.NLLLoss(ignore_index=self.pad_index,
                                         reduction='sum')
        else:
            self.criterion = nn.KLDivLoss(reduction='sum')

    def _smooth_target(self, targets: Tensor, vocab_size: int):
        """
        Smooth target distribution. All non-reference words get uniform
        probability mass according to "smoothing".

        :param targets: target indices, batch*seq_len
        :param vocab_size: size of the output vocabulary
        :return: smoothed target distributions, batch*seq_len x vocab_size
        """

        # batch * seq_len, vocab_size
        smooth_dict = targets.new_zeros((targets.size(0), vocab_size)).float()
        # fill distribution uniformly with smoothing, 1-pad, 1-true_label
        smooth_dict.fill_(self.smoothing / (vocab_size - 2))
        # assign true label the probability of 1-smoothing ("confidence")
        smooth_dict.scatter_(1, targets.unsqueeze(1).data, 1.0 - self.smoothing)
        # give padding probability of 0 everywhere
        smooth_dict[:, self.pad_index] = 0
        # masking out padding area (sum of probabilities for padding area = 0)
        # shape [zero_numbers, 1]
        padding_positions = torch.nonzero(targets.data == self.pad_index)
        if len(padding_positions) > 0:
            smooth_dict.index_fill_(0, padding_positions.squeeze(), 0.0)

        # [batch_size*seq_len, vocab_size]
        return Variable(smooth_dict, requires_grad=True)

    def forward(self, log_probs, targets):
        """
        Compute the cross-entropy between logits and targets.

        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.

        :param log_probs: log probabilities as predicted by model [batch_size, seq_len, vocab_size]
        :param target: target indices
        :return:
        """

        if self.smoothing > 0:
            # [batch_size*seq_len, vocab_size]
            targets = self._smooth_target(
                targets=targets.contiguous().view(-1),
                vocab_size=targets.size(-1)
            )

            # targets: distributions with batch*seq_len x vocab_size
            assert log_probs.contiguous().view(-1, log_probs.size(-1)).shape == targets.shape
        else:
            targets = targets.contiguous().view(-1)
        """
        KL情况下，target和log_probs相同维度  [batch*seq_len, vocab_size]
        NLL情况下, target: [batch_size*seq_len] log_probs: [batch*seq_len, vocab_size]
        """
        loss = self.criterion(
            log_probs.contiguous.view(-1, log_probs.size(-1)), targets
        )
        return loss
