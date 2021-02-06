import os, math, random
from argparse import ArgumentParser
from nltk.translate.bleu_score import corpus_bleu

import youtokentome as yttm
import pytorch_lightning as pl
import torch
from torchnlp.utils import lengths_to_mask

from models import transformer
from models import attention
from models import hard_concrete_gate

from datamodules.wmt import TransformerBatchedSequencesWithMasks
from pl_transformer import TransformerLightningModule

class PrunedEncoderTransformerLightningModule(TransformerLightningModule):
    def __init__(self,
        *args,
        **kwargs,
    ):

        super(PrunedEncoderTransformerLightningModule, self).__init__(*args, **kwargs)

        # enable hard concrete gates in encoder self attention
        # freeze decoder

        self.transformer: transformer.Transformer
        encoder_blocks_mhas = self.get_encoder_mha()

        hcg_l0_penalty_lambda = kwargs.get('hcg_l0_penalty_lambda', 0.02)
        print('hcg_l0_penalty_lambda', hcg_l0_penalty_lambda)

        for encoder_mha in encoder_blocks_mhas:
            encoder_mha.hard_concrete_gate = hard_concrete_gate.HardConcreteGate(encoder_mha.num_heads, l0_penalty_lambda=hcg_l0_penalty_lambda)

        return

    def get_encoder_mha(self) -> attention.SimpleMultiHeadAttention:
        return self.transformer.encoder_blocks.get_multihead_self_attention()

    def training_step(self, batch: TransformerBatchedSequencesWithMasks, batch_idx):

        self.transformer.decoder_blocks.eval() # freezes decoder parameters

        loss = super(PrunedEncoderTransformerLightningModule, self).training_step(batch, batch_idx)

        encoder_blocks_mhas = self.get_encoder_mha()
        for i, encoder_mha in enumerate(encoder_blocks_mhas):
            encoder_mha_l0_penalty = encoder_mha.hard_concrete_gate.l0_penalty.sum()
            self.log(f"{i}_encoder_mha_l0_sum_penalty", encoder_mha_l0_penalty.detach().cpu())

            # print("encoder_mha_l0_penalty", encoder_mha_l0_penalty)
            loss += encoder_mha_l0_penalty.sum()

        return loss


def cli_main(args=None):
    from datamodules.wmt import WMTDataModule

    pl.seed_everything()

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)

    # todo support other datamodules

    dm_class = WMTDataModule
    parser = PrunedEncoderTransformerLightningModule.add_model_specific_args(parser)
    parser = dm_class.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)

    dm = dm_class.from_argparse_args(args)
    dm.setup()

    if args.max_steps == -1:
        args.max_steps = None

    args.src_vocab_size = dm.src_bpe.vocab_size()
    args.trg_vocab_size = dm.trg_bpe.vocab_size()

    transformer_model = PrunedEncoderTransformerLightningModule.load_from_checkpoint(args.checkpoint, strict=False)
    transformer_model.trg_bpe = trg_bpe=dm.trg_bpe

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(transformer_model, datamodule=dm)
    return dm, transformer_model, trainer


if __name__ == "__main__":
    dm, model, trainer = cli_main()