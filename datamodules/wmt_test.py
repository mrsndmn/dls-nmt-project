import pytest

import datamodules.wmt as wmt

from datamodules.wmt import TransformerBatchedSequencesWithMasks

def test_wmt14_datamodule():
    max_lines = 10000
    batch_size = 5
    wmtdm = wmt.WMTDataModule(download=True, batch_size=batch_size, force=False, max_lines=max_lines)

    wmtdm.setup()

    triain_dl = wmtdm.train_dataloader()

    batch: TransformerBatchedSequencesWithMasks = next(iter(triain_dl))

    assert batch.src_tensor.size(0) == batch_size
    assert batch.trg_tensor.size(0) == batch_size
    assert batch.trg_tensor.size(1) == batch.src_tensor.size(1)
    assert batch.trg_y_tensor.size(1) == batch.src_tensor.size(1)

    # print(wmtdm.bad_lines)
    assert wmtdm.wmt is not None

    print("wmtdm.bad_lines", wmtdm.bad_lines)
