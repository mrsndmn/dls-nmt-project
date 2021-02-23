
import re
import io
import os
import typing
import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader, Dataset, random_split

import youtokentome as yttm
import pickle

from collections import namedtuple

TransformerBatchedSequencesWithMasks = namedtuple('TransformerBatchedSequencesWithMasks', ['src_tensor', 'src_mask', 'trg_tensor', 'trg_y_tensor', 'trg_mask', 'n_trg_tokens'])

class TextTranslationDataset(torch.utils.data.Dataset):
    def __init__(self, src_sentences, trg_sentences):
        """Initiate text-translation dataset.
        Args:
            todo add docs
        Examples:
            See the examples in examples/text_classification/
        """

        super(TextTranslationDataset, self).__init__()

        if len(src_sentences) != len(trg_sentences):
            raise ValueError(
                "source and target datasets must be the same length")

        self._src_sentences = src_sentences
        self._trg_sentences = trg_sentences

    def __getitem__(self, i):
        return self._src_sentences[-i], self._trg_sentences[-i]

    def __len__(self):
        return len(self._src_sentences)

    def __iter__(self):
        for x in zip(self._src_sentences, self._trg_sentences):
            yield x


class WMTDataModule(pl.LightningDataModule):
    """
    WMT19 datamodule
    """
    name = "wmt"

    def __init__(self,
                 src_bpe_tokenized_file: str,
                 trg_bpe_tokenized_file: str,
                 src_bpe_model_file: str = None,
                 trg_bpe_model_file: str = None,
                 src_vocab_size=10000,
                 trg_vocab_size=10000,
                 batch_size: int = 1,
                 val_batch_size: int = None,
                 max_lines=None,
                 max_seq_len_tokens=100,
                 min_seq_len_tokens=10,
                 valid_size = 3000,
                 ):
        super(WMTDataModule, self).__init__()

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size

        if src_bpe_tokenized_file is None:
            raise ValueError("src_bpe_tokenized_file is required")
        self.src_bpe_tokenized_file = src_bpe_tokenized_file
        if trg_bpe_tokenized_file is None:
            raise ValueError("trg_bpe_tokenized_file is required")
        self.trg_bpe_tokenized_file = trg_bpe_tokenized_file

        self.src_bpe_model_file = src_bpe_model_file
        self.trg_bpe_model_file = trg_bpe_model_file

        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size

        self.wmt = None

        self.bad_lines = 0
        self.max_lines = max_lines

        self.max_seq_len_tokens = max_seq_len_tokens
        self.min_seq_len_tokens = min_seq_len_tokens

        self.valid_size = valid_size

        return

    def _bpe_model(self, bpe_model_file):
        if bpe_model_file is None:
            raise ValueError("can't get bpe: bpe model file (argument `src_bpe_model_file|trg_bpe_model_file`) was not specified")

        return yttm.BPE(model=bpe_model_file)

    @property
    def src_bpe(self) -> yttm.BPE:
        return self._bpe_model(self.src_bpe_model_file)

    @property
    def trg_bpe(self) -> yttm.BPE:
        return self._bpe_model(self.trg_bpe_model_file)

    def _parse_bpe_tokenized_encoded_file(self, bpe_encoded_file):

        with open(bpe_encoded_file, 'r') as f:
            bpe_tokenized_sentences = []

            for line in f:
                line = line.strip()
                bpe_tokenized_sentences.append( list(map(int, line.split(" "))) )

            return bpe_tokenized_sentences

    def setup(self, stage=None):

        src_sentences = self._parse_bpe_tokenized_encoded_file(self.src_bpe_tokenized_file)
        trg_sentences = self._parse_bpe_tokenized_encoded_file(self.trg_bpe_tokenized_file)

        src_sentences_filtered = []
        trg_sentences_filtered = []
        for i in range(len(trg_sentences)):
            src = src_sentences[i]
            trg = trg_sentences[i]
            if len(src) > self.max_seq_len_tokens or len(src) < self.min_seq_len_tokens:
                continue
            if len(trg) > self.max_seq_len_tokens or len(trg) < self.min_seq_len_tokens:
                continue
            src_sentences_filtered.append(src)
            trg_sentences_filtered.append(trg)

        self.wmt = TextTranslationDataset(src_sentences, trg_sentences)

        train_len = len(self.wmt) - self.valid_size
        wmt_train, wmt_valid = torch.utils.data.random_split(self.wmt, [train_len, self.valid_size])
        self.wmt_train = wmt_train
        self.wmt_valid = wmt_valid

        return

    def collate_fn(self, batch: typing.List):
        src_tensors = []
        trg_tensors = []
        batch.sort(key=lambda x: len(x[1]))

        pad_idx = 0

        max_seq_len = 0
        for src_tokens, trg_tokens in batch:
            src_len = len(src_tokens)
            trg_len = len(trg_tokens)
            max_seq_len = max(max_seq_len, src_len, trg_len)

        src_max_seq_len = max_seq_len
        trg_max_seq_len = max_seq_len + 1

        # длинна src должна быть на один меньше, чем длинна таргета
        # потому что потом тагрет нужно будет сдвинуть
        batch_size = len(batch)
        src_padded = torch.full((batch_size, src_max_seq_len), pad_idx, dtype=torch.long)
        trg_padded = torch.full((batch_size, trg_max_seq_len), pad_idx, dtype=torch.long)

        for i, src_trg in enumerate(batch):
            src_tokens, trg_tokens = src_trg
            src_padded[i, :len(src_tokens)] = torch.LongTensor(src_tokens)
            trg_padded[i, :len(trg_tokens)] = torch.LongTensor(trg_tokens)

        trg_y_padded = trg_padded[:, 1:]
        trg_padded = trg_padded[:, :-1]

        src_mask = (src_padded != pad_idx).unsqueeze(-2)

        trui_tensor = (torch.triu(torch.ones(batch_size, max_seq_len, max_seq_len), diagonal=1) == 0)
        trg_mask = (trg_y_padded != pad_idx).unsqueeze(-2) & trui_tensor

        num_target_tokens = (trg_y_padded != pad_idx).sum()

        return TransformerBatchedSequencesWithMasks(src_padded, src_mask, trg_padded, trg_y_padded, trg_mask, num_target_tokens)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.wmt_train, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True, num_workers=1)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.wmt_valid, batch_size=self.val_batch_size, collate_fn=self.collate_fn, shuffle=False, num_workers=1)

