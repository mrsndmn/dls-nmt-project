
import re
import io
import os
import typing
import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchnlp.encoders.text.text_encoder import BatchedSequences

from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader

import torchtext
from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.utils import collate_tensors

import youtokentome as yttm
import pickle

from collections import namedtuple

TransformerBatchedSequencesWithMasks = namedtuple('TransformerBatchedSequencesWithMasks', ['src_tensor', 'src_mask', 'trg_tensor', 'trg_y_tensor', 'trg_mask', 'n_trg_tokens'])

class TextTranslationDataset(torch.utils.data.Dataset):
    def __init__(self, src_data, trg_data):
        """Initiate text-translation dataset.
        Args:
            todo add docs
        Examples:
            See the examples in examples/text_classification/
        """

        super(TextTranslationDataset, self).__init__()

        if len(src_data) != len(trg_data):
            raise ValueError(
                "source and target datasets must be the same length")

        self._src_data = src_data
        self._trg_data = trg_data

    def __getitem__(self, i):
        return self._src_data[i], self._trg_data[i]

    def __len__(self):
        return len(self._src_data)

    def __iter__(self):
        for x in zip(self._src_data, self._trg_data):
            yield x


class WMTDataModule(pl.LightningDataModule):
    """
    WMT19 datamodule
    """
    url = 'http://data.statmt.org/news-commentary/v14/training/news-commentary-v14.en-ru.tsv.gz'
    file_name = 'news-commentary-v14.en-ru.tsv'
    name = "wmt"

    def __init__(self,
                 batch_size: int = 1,
                 val_batch_size: int = 1,
                 download: bool = True,
                 root: str = '.data',
                 max_lines=None,
                 force=False,
                 src_vocab_size=10000,
                 trg_vocab_size=10000,
                 max_seq_len_tokens=100,
                 min_seq_len_tokens=10,
                 ):
        super(WMTDataModule, self).__init__()

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size

        self.download: bool = download
        self.data_root: str = root
        self.file_path: str = os.path.join(self.data_root, self.file_name)

        self.src_file = self.file_path + "_src_only"
        self.trg_file = self.file_path + "_trg_only"
        self.src_bpe_file = self.src_file + "_bpe"
        self.trg_bpe_file = self.trg_file + "_bpe"

        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size

        self.src_bpe: yttm.BPE = None
        self.trg_bpe: yttm.BPE = None

        self.wmt = None

        self.bad_lines = 0
        self.max_lines = max_lines

        self.force = force
        self.max_seq_len_tokens = max_seq_len_tokens
        self.min_seq_len_tokens = min_seq_len_tokens

        return

    def download_dataset(self):

        if not self.force and os.path.isfile(self.file_path):
            return

        dataset_tar = torchtext.utils.download_from_url(self.url)
        extracted_files = torchtext.utils.extract_archive(dataset_tar)
        assert len(extracted_files) == 1
        extracted_file = extracted_files[0]

        with io.open(os.path.expanduser(extracted_file), encoding="utf8") as f:
            file_lines = f.readlines()
        file_lines.sort(key=lambda x: len(x))

        self._remove_file(extracted_file)
        with io.open(os.path.expanduser(extracted_file), mode='w', encoding="utf8") as f:
            f.writelines(file_lines)

        # todo remove trash from file

        return extracted_file

    def preprocess_line(self, line: str):
        return line.lower()

    def split_files(self, file_path):

        if not self.force and os.path.isfile(self.src_file) and os.path.isfile(self.trg_file):
            return

        self._remove_file(self.src_file)
        self._remove_file(self.trg_file)

        with io.open(os.path.expanduser(file_path), encoding="utf8") as f:
            src_only_file = open(self.src_file, 'w')
            trg_only_file = open(self.trg_file, 'w')

            for i, line in enumerate(f):
                line = line.strip()
                line = line.split("\t")

                if len(line) != 2 or line[0] == "" or line[1] == "":
                    self.bad_lines += 1
                    continue

                if self.max_lines is not None and self.max_lines <= i:
                    break

                src_line: str
                trg_line: str
                src_line = self.preprocess_line(line[0])
                trg_line = self.preprocess_line(line[1])

                src_only_file.write(src_line + "\n")
                trg_only_file.write(trg_line + "\n")

            src_only_file.close()
            trg_only_file.close()

        return

    def _remove_file(self, file):
        if os.path.isfile(file):
            os.remove(file)
            return True
        return False

    def bpe_tokenize(self):
        self.split_files(self.file_path)

        if self.force or not os.path.isfile(self.src_bpe_file) or not os.path.isfile(self.src_bpe_file):

            self._remove_file(self.src_bpe_file)
            self._remove_file(self.trg_bpe_file)

            yttm.BPE.train(
                data=self.src_file, vocab_size=self.src_vocab_size, model=self.src_bpe_file)
            yttm.BPE.train(
                data=self.trg_file, vocab_size=self.trg_vocab_size, model=self.trg_bpe_file)

        # Loading model
        self.src_bpe: yttm.BPE = yttm.BPE(model=self.src_bpe_file)
        self.trg_bpe: yttm.BPE = yttm.BPE(model=self.trg_bpe_file)

        return

    def tokenize_file(self, file, tokenizer: yttm.BPE):
        data = []
        with io.open(os.path.expanduser(file), encoding="utf8") as f:
            for line in f:
                tokenized_line = tokenizer.encode(
                    line, output_type=yttm.OutputType.ID, bos=True, eos=True)
                data.append(tokenized_line)

        return data

    def setup(self, stage=None):

        self.download_dataset()

        self.bpe_tokenize()

        self.src_pickle = self.file_path + ".pickle"
        if os.path.isfile(self.src_pickle):
            with open(self.src_pickle, 'rb') as f:
                src_data = pickle.load(f)
                trg_data = pickle.load(f)
        else:
            with open(self.src_pickle, 'wb') as f:
                src_data = self.tokenize_file(self.src_file, self.src_bpe)
                trg_data = self.tokenize_file(self.trg_file, self.trg_bpe)
                src_data_filtered = []
                trg_data_filtered = []
                for i in range(len(trg_data)):
                    src = src_data[i]
                    trg = trg_data[i]
                    if len(src) > self.max_seq_len_tokens or len(src) < self.min_seq_len_tokens:
                        continue
                    if len(trg) > self.max_seq_len_tokens or len(trg) < self.min_seq_len_tokens:
                        continue
                    src_data_filtered.append(src)
                    trg_data_filtered.append(trg)

                pickle.dump(src_data_filtered, f)
                pickle.dump(trg_data_filtered, f)


        self.wmt = TextTranslationDataset(src_data, trg_data)

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

        src_mask = (src_padded != pad_idx)

        trui_tensor = (torch.triu(torch.ones(batch_size, max_seq_len, max_seq_len), diagonal=1) == 0)
        trg_mask = (trg_y_padded != pad_idx).unsqueeze(-2) & trui_tensor

        num_target_tokens = (trg_y_padded != pad_idx).sum()

        return TransformerBatchedSequencesWithMasks(src_padded, src_mask, trg_padded, trg_y_padded, trg_mask, num_target_tokens)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.wmt, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=False, num_workers=1)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.wmt, batch_size=self.val_batch_size, collate_fn=self.collate_fn, shuffle=False, num_workers=1)

