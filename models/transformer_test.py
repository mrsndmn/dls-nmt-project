import pytest

from models.transformer import Transformer, TransformerGenerator

import torch

def test_transformer_forward():
    with torch.no_grad():

        src_vocab_size = 10
        trg_vocab_size = 10

        hid_dim = 64
        kqv_dim = 4

        tfm = Transformer(src_vocab_size, trg_vocab_size, hid_dim, key_query_value_dim=kqv_dim, num_heads=16)

        batch_size = 3
        seq_len = 7

        src_tokens = torch.randint(0, src_vocab_size, (batch_size, seq_len))
        trg_tokens = torch.randint(0, trg_vocab_size, (batch_size, seq_len))

        trnsf_output = tfm.forward(src_tokens, trg_tokens)
        assert trnsf_output.shape == torch.Size((batch_size, seq_len, hid_dim))


def test_generator():
    with torch.no_grad():
        hid_dim = 64
        trg_vocab_size = 10
        trnsfm_gen = TransformerGenerator(hid_dim, trg_vocab_size)

        batch_size = 3
        seq_len = 7
        trnsfm_gen_input = torch.rand((batch_size, seq_len, hid_dim))
        trnsfm_gen_output = trnsfm_gen.forward(trnsfm_gen_input)
        assert trnsfm_gen_output.size() == torch.Size((batch_size, seq_len, trg_vocab_size))