import pytest

from models.decoder import TransformerDecoderBlock, TransformerEncoderDecoderConnectionBlock

import torch

def test_decoder_block():
    with torch.no_grad():
        hid_dim = 64
        kq_dim= 4
        decoder_block = TransformerDecoderBlock(hid_dim, key_query_value_dim=kq_dim, num_attention_heads=16, feed_forward_hidden_dim=128)

        batch_size = 3
        seq_len = 7
        decoder_hidden = torch.rand((batch_size, seq_len, hid_dim))
        encoder_outputs_hidden = torch.rand((batch_size, seq_len, hid_dim))

        decoder_output = decoder_block.forward(decoder_hidden, encoder_outputs_hidden)
        assert decoder_output.size() == torch.Size((batch_size, seq_len, hid_dim))

def test_encoder_decoder_block():
    with torch.no_grad():
        hid_dim = 64
        kqv_dim = 4
        num_heads = hid_dim // kqv_dim

        batch_size = 3
        seq_len = 7
        encoder_outputs = torch.rand((batch_size, seq_len, hid_dim))
        decoder_hidden = torch.rand((batch_size, seq_len, hid_dim))

        encoder_decoder_block = TransformerEncoderDecoderConnectionBlock(hid_dim, key_query_value_dim=kqv_dim, num_attention_heads=num_heads)
        decoder_output = encoder_decoder_block.forward(encoder_outputs, decoder_hidden)
        assert decoder_output.size() == torch.Size((batch_size, seq_len, hid_dim))