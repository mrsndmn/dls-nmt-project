import pytest

import datamodules.wmt as wmt
from nltk.tokenize import WordPunctTokenizer

from torchnlp.encoders.text.text_encoder import BatchedSequences

def test_wmt14_datamodule():
    wpt = WordPunctTokenizer()
    max_lines = 10000
    batch_size = 5
    wmtdm = wmt.WMTDataModule(wpt, wpt, download=True, batch_size=batch_size, force=True, max_lines=max_lines)

    wmtdm.setup()

    triain_dl = wmtdm.train_dataloader()

    src_padded_tokens: BatchedSequences
    trg_padded_tokens: BatchedSequences
    src_padded_tokens, trg_padded_tokens = next(iter(triain_dl))

    print("src_padded_tokens", src_padded_tokens.tensor.shape)
    print("trg_padded_tokens", trg_padded_tokens.tensor.shape)

    print(wmtdm.src_bpe.decode(list(src_padded_tokens.tensor.numpy())))
    print(wmtdm.trg_bpe.decode(list(trg_padded_tokens.tensor.numpy())))

    assert len(src_padded_tokens.tensor.shape) == 2 # batch_size x max_seq_len
    assert src_padded_tokens.tensor.shape[0] == batch_size
    assert trg_padded_tokens.tensor.shape[0] == batch_size

    print(wmtdm.bad_lines)
    assert wmtdm.wmt is not None
    assert 0 < len(wmtdm.wmt) < max_lines

    print("wmtdm.bad_lines", wmtdm.bad_lines)
