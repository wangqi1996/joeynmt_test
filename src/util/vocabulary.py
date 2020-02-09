# coding=utf-8

"""
vocabulary model
word和id的映射
"""
from collections import defaultdict, Counter
from typing import List

import numpy as np
from torchtext.data import Dataset

from src.util.constants import UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN, UNK_ID, DEFAULT_VOCAB_VALUE


class Vocabulary:
    """ Vocabulary represents mapping between tokens and indices. """

    def __init__(self, tokens: List[str] = None, file: str = None) -> None:
        """

        Create vocabulary from list of tokens or file.
        Special tokens are added if not already in file or list.
        File format: token with index i is in line i.

        """
        assert tokens is not None and file is not None, u'only can init vocab from tokens or file'
        # special symbols
        self.specials = [UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, PAD_TOKEN]

        # 构造一个词典，默认值为0
        self.stoi = defaultdict(DEFAULT_VOCAB_VALUE)
        self.itos = []  # 用数组下标对齐

        if tokens is not None:
            self._from_list(tokens)
        elif file is not None:
            self._from_file(file)

    def _from_list(self, tokens: List[str] = None) -> None:
        """
       Make vocabulary from list of tokens.

       Tokens are assumed to be unique and pre-selected.
       Special symbols are added if not in list.
       """

        self.add_tokens(tokens=tokens + self.specials)
        assert len(self.itos) == len(self.stoi)

    def add_tokens(self, tokens):
        """

        add a list tokens to vocabulary
        """
        for t in tokens:
            new_index = len(self.itos)

            # add to vocab is not already there
            if t not in self.itos:
                self.itos.append(t)
                self.stoi[t] = new_index

    def _from_file(self, file: str) -> None:
        """
        Make vocabulary from contents of file.
        File format: token with index i is in line i.

        :param file: vocab file-name, contain special tokens
        :return:
        """

        tokens = []
        with open(file, 'r') as open_file:
            for line in open_file:
                tokens.append(line.strip('\n'))

        self._from_list(tokens)

    def __str__(self) -> str:
        return self.stoi.__str__()

    def to_file(self, file: str) -> None:
        """
        Save the vocabulary to a file, by writing token with index i in line i

        :param file:
        :return:
        """
        with open(file, 'w') as open_file:
            for t in self.itos:
                open_file.write("{}\n".format(t))

    def is_unk(self, token: str) -> bool:
        """
        check whether a token is covered by the vocabulary

        :param token:
        :return:
        """

        return self.stoi[token] == UNK_ID

    def __len__(self) -> int:
        return len(self.itos)

    def array_to_sentence(self, array: np.array, cut_at_eos=True):
        """
        convert an array of IDs to a sentence, optional cutting the result
        off at the end-of-sequence token.

        :param array:   1D array containing indices
        :param cut_at_eos:   cut the decoded sentences at the first <eos>
        :return:  list of strings (tokens)
        """

        sentence = []

        for i in array:
            s = self.itos[i]
            if cut_at_eos and s == EOS_TOKEN:
                break
            sentence.append(s)

        return sentence

    def array_to_sentences(self, arrays: np.array, cut_at_eos=True) -> List[List[str]]:
        """
        Convert multiple arrays containing sequences of token IDs to their
        sentences, optionally cutting them off at the end-of-sequence token.

        :param arrays: 2D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :return: list of list of strings (tokens)
        """
        sentences = []
        for array in arrays:
            sentences.append(
                self.array_to_sentence(array, cut_at_eos=cut_at_eos)
            )

        return sentences


def build_vocab(field: str, max_size: int, min_freq: int, dataset: Dataset,
                vocab_file: str = None) -> Vocabulary:
    """
    Builds vocabulary for a torchtext `field` from given`dataset` or
    `vocab_file`.

    :param field: attribute e.g. "src" or "trg"
    :param max_size: maximum size of vocabulary
    :param min_freq: minimum frequency for an item to be included
    :param dataset: dataset to load data for field from
    :param vocab_file: file to store the vocabulary,
        if not None, load vocabulary from here

    :return: Vocabulary created from either `dataset` or `vocab_file`
    """
    if vocab_file is not None:
        vocab = Vocabulary(file=vocab_file)
    else:
        # create newly

        def filter_min(counter: Counter, limit: int):
            """
            filter counter by min frequency
            """
            filter_counter = Counter(
                {t: c for t, c in counter.items() if c >= min_freq}
            )

            return filter_counter

        def sort_and_cut(counter: Counter, limit: int):
            """
            cut counter to most frequent, sorted numerically and alphabetically
            """
            # sort by frequency, then alphabetically
            tokens_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
            tokens_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
            vocab_tokens = [i[0] for i in tokens_and_frequencies[:limit]]

            return vocab_tokens

        tokens = []
        for i in dataset.examples:
            if field == 'src':
                tokens.extend(i.src)
            elif field == 'trg':
                tokens.extend(i.trg)

        counter = Counter(tokens)

        if min_freq > -1:
            counter = filter_min(counter, min_freq)

        vocab_tokens = sort_and_cut(counter, max_size)
        assert len(vocab_tokens) <= max_size

        vocab = Vocabulary(tokens=vocab_tokens)
        assert len(vocab) <= max_size + len(vocab.specials)
        assert vocab.itos[UNK_ID] == UNK_TOKEN

    for s in vocab.specials[1:]:
        assert not vocab.is_unk(s)

    return vocab
