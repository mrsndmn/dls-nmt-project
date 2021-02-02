import os, math, random
from argparse import ArgumentParser
from nltk.translate.bleu_score import corpus_bleu

import youtokentome as yttm
import pytorch_lightning as pl
import torch
from torchnlp.utils import lengths_to_mask

from models import transformer
from datamodules.wmt import TransformerBatchedSequencesWithMasks


class TransformerLightningModule(pl.LightningModule):
    def __init__(self,
        src_vocab_size: int,
        trg_vocab_size: int,
        hidden_dim: int=256,
        num_blocks: int=6,
        key_query_value_dim: int=32,
        padding_token_idx: int = 0,
        smoothing: float = 0.1,
        lr: float = 1e-4,
        noam_opt_warmup_steps: int= 4000,
        trg_bpe=None,
    ):

        super(TransformerLightningModule, self).__init__()

        self.save_hyperparameters("src_vocab_size", "trg_vocab_size", "hidden_dim", "num_blocks", "key_query_value_dim", "padding_token_idx", "smoothing", "lr", "noam_opt_warmup_steps")

        self.transformer = transformer.Transformer(src_vocab_size, trg_vocab_size, hidden_dim,
                                                   num_blocks=num_blocks,
                                                   key_query_value_dim=key_query_value_dim)

        self.criterion = transformer.LabelSmoothing(trg_vocab_size, padding_token_idx=padding_token_idx, smoothing=smoothing)

        self.trg_bpe: yttm.BPE = trg_bpe

        return

    def training_step(self, batch: TransformerBatchedSequencesWithMasks, batch_idx):

        transformer_output = self.transformer.forward(batch.src_tensor, batch.trg_tensor, src_mask=batch.src_mask, trg_mask=batch.trg_mask)
        trg_tokens_probabilities = self.transformer.generator.forward(transformer_output) # batch_size, seq_len, hidd_dim

        loss = self.criterion(trg_tokens_probabilities, batch.trg_y_tensor)
        loss /= batch.n_trg_tokens

        self.log("loss", loss.item())

        return loss

    def _filter_eos_seq(self, sentenses, eos_tok_id=3):
        sentenses_decoded = []
        for _seq in sentenses:
            seq = []
            for tok in _seq:
                if tok == 3:
                    break
                seq.append(tok)
            sentenses_decoded.append(seq)
        return sentenses_decoded

    # todo dropout
    def validation_step(self, batch: TransformerBatchedSequencesWithMasks, batch_idx: int):

        translation = self.transformer.encode_decode(batch.src_tensor, src_mask=batch.src_mask)

        ignore_ids = [0,1,2]
        translation = translation.detach().cpu().numpy().tolist()
        translation = self._filter_eos_seq(translation)
        translation_decoded = self.trg_bpe.decode(translation, ignore_ids=ignore_ids)

        target = batch.trg_tensor.detach().cpu().numpy().tolist()
        target = self._filter_eos_seq(target)
        target_decoded = self.trg_bpe.decode(target, ignore_ids=ignore_ids)

        assert len(translation_decoded) == len(target_decoded)

        return translation_decoded, target_decoded

    def validation_epoch_end(self, validation_step_outputs):
        generated = []
        references = []

        for vout in validation_step_outputs:
            for gen_seq in vout[0]:
                generated.append(gen_seq)
            for trg_seq in vout[1]:
                references.append([trg_seq])

        translation_str = "\n\n\n".join(generated[:5])
        target_str = "\n\n\n".join(x[0] for x in references[:5])
        self.logger.experiment.add_text("translate_decoded", translation_str)
        self.logger.experiment.add_text("translate_target", target_str)

        calculated_bleu = corpus_bleu(references, generated)
        self.log("valid_bleu", calculated_bleu)
        return

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--src_vocab_size", type=int, default=10000)
        parser.add_argument("--trg_vocab_size", type=int, default=10000)
        parser.add_argument("--hidden_dim", type=int)
        parser.add_argument("--num_blocks", type=int)
        parser.add_argument("--key_query_value_dim", type=int)

        parser.add_argument("--lr", type=float)

        # parser.add_argument("--num_workers", type=int, default=8)
        # parser.add_argument("--data_dir", type=str, default=".")

        return parser

    def noam_opt(self, current_step: int):
        min_inv_sqrt = min(1/math.sqrt(self.trainer.global_step+1), self.trainer.global_step * self.hparams.noam_opt_warmup_steps ** (-1.5))
        current_lr = 2 * min_inv_sqrt / math.sqrt(self.hparams.hidden_dim)

        self.logger.experiment.add_scalar("lr", current_lr, self.trainer.global_step)
        return current_lr

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.transformer.parameters(), lr=self.hparams.lr)
        opt_sched = torch.optim.lr_scheduler.LambdaLR(opt, self.noam_opt)
        return [opt], [{"scheduler": opt_sched, "interval": "step"}]

# copy-paste https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py
def cli_main(args=None):
    from datamodules.wmt import WMTDataModule

    pl.seed_everything()

    parser = ArgumentParser()

    # todo support other datamodules

    dm_class = WMTDataModule
    parser = TransformerLightningModule.add_model_specific_args(parser)
    parser = dm_class.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)

    dm = dm_class.from_argparse_args(args)
    dm.setup()

    if args.max_steps == -1:
        args.max_steps = None

    args.src_vocab_size = dm.src_bpe.vocab_size()
    args.trg_vocab_size = dm.trg_bpe.vocab_size()

    transformer_model = TransformerLightningModule(args.src_vocab_size, args.trg_vocab_size,
                                                   hidden_dim=args.hidden_dim,
                                                   num_blocks=args.num_blocks,
                                                   key_query_value_dim=args.key_query_value_dim,
                                                   trg_bpe=dm.trg_bpe)

    print("training")
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(transformer_model, datamodule=dm)
    return dm, transformer_model, trainer


if __name__ == "__main__":
    dm, model, trainer = cli_main()