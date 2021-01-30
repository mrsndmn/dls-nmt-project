import torch
import torch.nn as nn

from models.attention import AddAndNorm, MultiHeadAttention, SimpleMultiHeadAttention

from typing import List

class TransformerEncoderDecoderConnectionBlock(nn.Module):
    def __init__(self, hidden_dim: int, key_query_value_dim: int = 64, num_attention_heads=8):
        super(TransformerEncoderDecoderConnectionBlock, self).__init__()

        self.multihead_attention = MultiHeadAttention(hidden_dim, key_and_query_dim=key_query_value_dim, value_dim=key_query_value_dim, num_heads=num_attention_heads)
        self.norm = nn.LayerNorm(hidden_dim)
        return

    def forward(self, encoder_outputs, decoder_hidden, mask=None):
        attention_outputs = self.multihead_attention.forward(k_hidden_inputs=encoder_outputs, q_hidden_inputs=encoder_outputs, v_hidden_inputs=decoder_hidden, mask=mask)
        return self.norm(decoder_hidden + attention_outputs)

class TransformerDecoderBlock(nn.Module):
    '''
    param: hidden_dim - embedding hidden dim
    '''

    def __init__(self, hidden_dim: int, key_query_value_dim: int = 64, num_heads=8, feed_forward_hidden_dim: int = 2048):
        super(TransformerDecoderBlock, self).__init__()

        masked_multihead_attention = SimpleMultiHeadAttention(
            hidden_dim, key_query_value_dim=key_query_value_dim, num_heads=num_heads)

        # todo move the same block in encode to separate class
        feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, feed_forward_hidden_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_hidden_dim, hidden_dim),
        )

        self.masked_multihead_attention_add_norm = AddAndNorm(
            hidden_dim, masked_multihead_attention)

        self.encoder_decoder_block = TransformerEncoderDecoderConnectionBlock(hidden_dim, key_query_value_dim=key_query_value_dim, num_attention_heads=num_heads)
        self.encoder_decoder_norm = nn.LayerNorm(hidden_dim)

        self.feed_forward_add_norm = AddAndNorm(hidden_dim, feed_forward)

        return

    def forward(self, decoder_hidden, encoder_outputs, src_mask=None, trg_mask=None):

        decoder_hidden = self.masked_multihead_attention_add_norm(
            decoder_hidden, mask=trg_mask)
        encoder_decoder_outputs = self.encoder_decoder_block(
            encoder_outputs, decoder_hidden, mask=trg_mask)

        feed_forward_inputs = self.encoder_decoder_norm(
            decoder_hidden + encoder_decoder_outputs)

        return self.feed_forward_add_norm(feed_forward_inputs)


class DecoderBlocksSequential(nn.Module):
    def __init__(self, decoder_blocks: List[TransformerDecoderBlock]):
        super(DecoderBlocksSequential, self).__init__()
        self.decoder_blocks = nn.ModuleList(decoder_blocks)

    def forward(self, target_embeddings, encoder_outputs, src_mask=None, trg_mask=None):
        decoder_output = target_embeddings
        for decoder_block in self.decoder_blocks:
            decoder_output = decoder_block(decoder_output, encoder_outputs, src_mask=src_mask, trg_mask=trg_mask)
        return decoder_output
