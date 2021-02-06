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

        self.hcg_l0_penalty_lambda = kwargs.get('hcg_l0_penalty_lambda', 0.02)

        self.valid_img_step = 0

        return

    def setup_encoder_hcg(self):
        encoder_blocks_mhas = self.get_encoder_mha()

        hcg_l0_penalty_lambda = self.hcg_l0_penalty_lambda
        print('hcg_l0_penalty_lambda', hcg_l0_penalty_lambda)

        for encoder_mha in encoder_blocks_mhas:
            encoder_mha.hard_concrete_gate = hard_concrete_gate.HardConcreteGate(encoder_mha.num_heads, l0_penalty_lambda=hcg_l0_penalty_lambda)


    def get_encoder_mha(self) -> attention.SimpleMultiHeadAttention:
        return self.transformer.encoder_blocks.get_multihead_self_attention()

    def training_step(self, batch: TransformerBatchedSequencesWithMasks, batch_idx):

        self.transformer.decoder_blocks.eval() # freezes decoder parameters

        loss = super(PrunedEncoderTransformerLightningModule, self).training_step(batch, batch_idx)

        encoder_blocks_mhas = self.get_encoder_mha()
        for i, encoder_mha in enumerate(encoder_blocks_mhas):
            encoder_mha_l0_penalty = encoder_mha.hard_concrete_gate.l0_penalty.sum()
            self.log(f"{i}_encoder_mha_l0_sum_penalty", encoder_mha_l0_penalty.detach().cpu())

            loss += encoder_mha_l0_penalty

        return loss

    def validation_step(self, batch: TransformerBatchedSequencesWithMasks, batch_idx: int):
        super_validation_step_outputs = super(PrunedEncoderTransformerLightningModule, self).validation_step(batch, batch_idx)
        encoder_blocks_mhas = self.get_encoder_mha()

        encoders_mha_l0_penalties = []
        for i, encoder_mha in enumerate(encoder_blocks_mhas):
            encoder_mha_l0_penalty = encoder_mha.hard_concrete_gate.l0_penalty
            encoders_mha_l0_penalties.append(encoder_mha_l0_penalty.unsqueeze(0))

        # num_blocks x num_heads
        encoders_mha_l0_penalties = torch.cat(encoders_mha_l0_penalties, dim=0)
        return super_validation_step_outputs, encoders_mha_l0_penalties

    def validation_epoch_end(self, validation_step_outputs):
        validation_step_outputs_super = []
        encoders_mha_l0_penalties = []
        for x in validation_step_outputs:
            validation_step_outputs_super.append(x[0])
            encoders_mha_l0_penalties.append(x[1].unsqueeze(0))

        super(PrunedEncoderTransformerLightningModule, self).validation_epoch_end(validation_step_outputs_super)

        encoders_mha_l0_penalties = torch.cat(encoders_mha_l0_penalties, dim=0)

        # valid_steps x num_blocks x num_heads
        encoders_mha_l0_penalties = encoders_mha_l0_penalties.mean(dim=0)
        encoders_mha_l0_penalties.unsqueeze_(0)

        # 1, num_blocks x num_heads
        encoders_mha_l0_penalties = encoders_mha_l0_penalties.detach().cpu()
        encoders_mha_l0_penalties = torch.repeat_interleave(encoders_mha_l0_penalties, 100, dim=1)
        encoders_mha_l0_penalties = torch.repeat_interleave(encoders_mha_l0_penalties, 100, dim=2)

        self.logger.experiment.add_image('encoders_attentions', encoders_mha_l0_penalties, self.valid_img_step)
        self.valid_img_step += 1

        return


def cli_main(args=None):
    from datamodules.wmt import WMTDataModule

    pl.seed_everything()

    parser = ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--hcg_l0_penalty_lambda", required=True, type=float)

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
    transformer_model.hparams.lr = args.lr
    transformer_model.hparams.scheduler=args.scheduler
    transformer_model.hparams.scheduler_patience=args.scheduler_patience
    transformer_model.hparams.noam_step_factor=args.noam_step_factor

    transformer_model.hcg_l0_penalty_lambda = args.hcg_l0_penalty_lambda
    transformer_model.setup_encoder_hcg()
    print("transformer_model.hparams", transformer_model.hparams, "hcg_l0_penalty_lambda", transformer_model.hcg_l0_penalty_lambda)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(transformer_model, datamodule=dm)
    return dm, transformer_model, trainer


if __name__ == "__main__":
    dm, model, trainer = cli_main()