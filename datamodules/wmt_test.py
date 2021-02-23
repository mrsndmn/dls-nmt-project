import pytest

import datamodules.wmt as wmt

def test_wmt14_datamodule():
    max_lines = 10000
    batch_size = 5

    src_bpe_tokenized_file = '.data/SRC_BPE_ENCODED_news-commentary-v14.en-ru.tsv'
    trg_bpe_tokenized_file = '.data/TRG_BPE_ENCODED_news-commentary-v14.en-ru.tsv'
    wmtdm = wmt.WMTDataModule(src_bpe_tokenized_file=src_bpe_tokenized_file,
                              trg_bpe_tokenized_file=trg_bpe_tokenized_file,
                              batch_size=batch_size)

    wmtdm.setup()

    triain_dl = wmtdm.train_dataloader()

    batch: wmt.TransformerBatchedSequencesWithMasks = next(iter(triain_dl))

    assert batch.src_tensor.size(0) == batch_size
    assert batch.trg_tensor.size(0) == batch_size
    assert batch.trg_tensor.size(1) == batch.src_tensor.size(1)
    assert batch.trg_y_tensor.size(1) == batch.src_tensor.size(1)

    assert wmtdm.wmt is not None

    print("wmtdm.bad_lines", wmtdm.bad_lines)
