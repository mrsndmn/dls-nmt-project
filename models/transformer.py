import math

import torch
import torch.nn as nn

from models.encoder import TransformerEncoderBlock, EncoderBlocksSequential
from models.decoder import TransformerDecoderBlock, DecoderBlocksSequential


# copy-paste from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, trg_vocab_size: int, hidden_dim: int, num_blocks: int = 6, key_query_value_dim=64, num_heads: int = None):
        super(Transformer, self).__init__()

        if num_heads is None:
            num_heads = hidden_dim // key_query_value_dim

        if num_heads <= 0:
            raise ValueError('num_heads must be positive')

        self.input_embeddings = nn.Sequential(
            nn.Embedding(src_vocab_size, hidden_dim),
            PositionalEncoding(hidden_dim),
        )
        self.target_embeddings = nn.Sequential(
            nn.Embedding(trg_vocab_size, hidden_dim),
            PositionalEncoding(hidden_dim),
        )

        self.num_blocks = num_blocks

        encoder_blocks = [TransformerEncoderBlock(
            hidden_dim, key_query_value_dim=key_query_value_dim, num_heads=num_heads) for _ in range(num_blocks)]
        decoder_blocks = [TransformerDecoderBlock(
            hidden_dim, key_query_value_dim=key_query_value_dim, num_heads=num_heads) for _ in range(num_blocks)]

        self.encoder_blocks = EncoderBlocksSequential(encoder_blocks)
        self.decoder_blocks = DecoderBlocksSequential(decoder_blocks)

        self.generator = TransformerGenerator(hidden_dim, trg_vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_tokens, trg_tokens, src_mask=None, trg_mask=None):
        # todo можно обучать так, чтобы один выход энкодера соответствовал
        # нескольким прогонам в декодере, на каждый токен
        # так данные будут более эффективно исползованы и уменьшится кол-во вызовов энкодера

        src_embeddings = self.input_embeddings(src_tokens)
        trg_embeddings = self.target_embeddings(trg_tokens)

        encoder_outputs = self.encoder_blocks.forward(
            src_embeddings, src_mask=src_mask)
        decoder_output = self.decoder_blocks.forward(
            trg_embeddings, encoder_outputs, src_mask=src_mask, trg_mask=trg_mask)

        return decoder_output


class TransformerGenerator(nn.Module):
    def __init__(self, hidden_dim, trg_vocab_size):
        super(TransformerGenerator, self).__init__()

        self.out_linear = nn.Sequential(
            nn.Linear(hidden_dim, trg_vocab_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, transformer_output):
        last_token = transformer_output[:, -1, :] # batch_size, hidden_dim
        return self.out_linear(last_token)


# copy-paste http://nlp.seas.harvard.edu/2018/04/03/attention.html#decoder
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, trg_vocab_size, padding_token_idx: int = 0, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_token_idx = padding_token_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.trg_vocab_size = trg_vocab_size
        self.true_dist = None

    def forward(self, trg_tokens_probas, target_token_idxs):
        assert trg_tokens_probas.size(1) == self.trg_vocab_size
        true_dist = trg_tokens_probas.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target_token_idxs.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_token_idx] = 0
        mask = torch.nonzero(target_token_idxs.data == self.padding_token_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(trg_tokens_probas, torch.Variable(true_dist, requires_grad=False))
