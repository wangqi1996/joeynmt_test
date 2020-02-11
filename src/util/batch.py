# coding=utf-8


class Batch:
    """
    Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(self, torch_batch, pad_index, use_cuda=False):
        """
        Create a new joey batch from a torch batch.
        This batch extends torch text's batch attributes with src and trg
        length, masks, number of non-padded tokens in trg.
        Furthermore, it can be sorted by src length.

        :param torch_batch:
        :param pad_index:
        :param use_cuda:
        """
        # [batch_size, seq_len]
        self.src, self.src_length = torch_batch.src

        # [batch_size, 1, seq_len]
        self.src_mask = (self.src != pad_index).unsqueeze(1)

        # batch_size
        self.nseqs = self.src.size(0)

        # trg
        self.trg_input = None
        self.trg = None
        self.trg_lengths = None
        self.trg_mask = None
        self.ntokens = None

        self.use_cuda = use_cuda

        if hasattr(torch_batch, 'trg'):
            trg, trg_lengths = torch_batch.trg
            # trg_input is used for teacher forcing, last one is cut off
            self.trg_input = trg[:, :-1]  # 去掉了最后一个词
            self.trg_lengths = trg_lengths
            # trg is used for loss computation, shifted by one since BOS
            self.trg = trg[:, 1:]  # 去掉第一个词
            # we exclude the padded areas from the loss computation
            # [batch_size, 1, seq_len]
            self.trg_mask = (self.trg_input != pad_index).unsqueeze(1)
            # [batch_size, seq_len] 全部的数据加和
            self.ntokens = (self.trg != pad_index).data.sum().item()

        if use_cuda:
            self._make_cuda()

    def _make_cuda(self):
        """
        move the batch to gpr

        """
        self.src = self.src.cuda()
        self.src_mask = self.src_mask.cuda()

        if self.trg_input is not None:
            self.trg.cuda()
            self.trg_input.cuda()
            self.trg_mask.cuda()

    def sort_by_src_length(self):
        """
        sort by src length (descending) and return index to revert sort
        :return:
        """
        _, perm_index = self.src_length.sort(0, descending=True)
        rev_index = [0] * perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpu()):
            rev_index[old_pos] = new_pos

        sorted_src_lengths = self.src_length[perm_index]
        sorted_src = self.src[perm_index]
        sorted_src_mask = self.src_mask[perm_index]

        self.src = sorted_src
        self.src_mask = sorted_src_mask
        self.src_length = sorted_src_lengths

        if self.trg_input is not None:
            sorted_trg_input = self.trg_input[perm_index]
            sorted_trg = self.trg[perm_index]
            sorted_trg_mask = self.trg[perm_index]
            sorted_trg_length = self.trg_lengths[perm_index]

            self.trg_input = sorted_trg_input
            self.trg_mask = sorted_trg_mask
            self.trg = sorted_trg
            self.trg_lengths = sorted_trg_length

        if self.use_cuda:
            self._make_cuda()

        return rev_index
