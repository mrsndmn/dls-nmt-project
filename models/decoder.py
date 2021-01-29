import torch
import torch.nn as nn

from models.attention import AddAndNorm, MultiHeadAttention

class TransformerEncoderDecoderConnectionBlock(nn.Module):
    pass

class TransformerDecoderBlock(nn.Module):
    '''
    param: hidden_dim - embedding hidden dim
    '''
    def __init__(self, hidden_dim: int, key_query_value_dim: int = 64, num_attention_heads=8, feed_forward_hidden_dim:int=2048, num_heads=8):
        super(TransformerDecoderBlock, self).__init__()

        masked_multihead_attention = MaskedMultiHeadAttention(hidden_dim, key_query_value_dim=key_query_value_dim, value_dim=key_query_value_dim, num_heads=num_heads)

        feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, feed_forward_hidden_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_hidden_dim, hidden_dim),
        )

        self.masked_multihead_attention_add_norm = AddAndNorm(hidden_dim, masked_multihead_attention)

        self.encoder_decoder_block = TransformerEncoderDecoderConnectionBlock()
        self.encoder_decoder_norm = nn.LayerNorm(hidden_dim)

        self.feed_forward_add_norm = AddAndNorm(hidden_dim, feed_forward)

        return

    def forward(self, decoder_hidden, encoder_outputs):

        decoder_hidden = self.masked_multihead_attention_add_norm(decoder_hidden)
        encoder_decoder_outputs = self.encoder_decoder_block(encoder_outputs, decoder_hidden)

        feed_forward_inputs = self.encoder_decoder_norm(decoder_hidden + encoder_decoder_outputs)

        return self.feed_forward_add_norm(feed_forward_inputs)
