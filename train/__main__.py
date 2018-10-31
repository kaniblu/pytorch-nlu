import os
import logging
import tempfile
import argparse

import numpy as np
import torch
import torchmodels
import torch.optim as op
import torch.utils.data as td

import utils
import models
import models.jlu
import dataset
import evaluate
import inference
from . import embeds

MODES = ["word", "label", "intent"]

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument("--debug", action="store_true", default=False)

group = parser.add_argument_group("Logging Options")
utils.add_logging_arguments(group, "train")
group.add_argument("--show-progress", action="store_true", default=False)

group = parser.add_argument_group("Model Options")
group.add_argument("--model-path", required=True)
group.add_argument("--hidden-dim", type=int, default=300)
group.add_argument("--word-dim", type=int, default=300)

group = parser.add_argument_group("Data Options")
for mode in MODES:
    group.add_argument(f"--{mode}-path", type=str, required=True)
    group.add_argument(f"--{mode}-vocab", type=str, default=None)
group.add_argument("--vocab-limit", type=int, default=None)
group.add_argument("--data-workers", type=int, default=8)
group.add_argument("--pin-memory", action="store_true", default=False)
group.add_argument("--shuffle", action="store_true", default=False)
group.add_argument("--max-length", type=int, default=None)
group.add_argument("--seed", type=int, default=None)
group.add_argument("--unk", type=str, default="<unk>")
group.add_argument("--eos", type=str, default="<eos>")
group.add_argument("--bos", type=str, default="<bos>")

group = parser.add_argument_group("Training Options")
group.add_argument("--save-dir", type=str, required=True)
group.add_argument("--save-period", type=int, default=None)
group.add_argument("--batch-size", type=int, default=32)
group.add_argument("--epochs", type=int, default=12)
group.add_argument("--optimizer", type=str, default="adam",
                   choices=["adam", "adamax", "adagrad", "adadelta", "sgd"])
group.add_argument("--loss-alpha", type=float, default=1.0)
group.add_argument("--loss-beta", type=float, default=1.0)
group.add_argument("--early-stop", action="store_true", default=False)
group.add_argument("--early-stop-patience", type=int, default=5)
group.add_argument("--early-stop-save", action="store_true", default=False)
group.add_argument("--early-stop-criterion", default="acc-label")
group.add_argument("--learning-rate", type=float, default=None)
group.add_argument("--weight-decay", type=float, default=None)
group.add_argument("--samples", type=int, default=None)
group.add_argument("--log-stats", action="store_true", default=False)
group.add_argument("--tensorboard", action="store_true", default=False)
group.add_argument("--resume-from")
group.add_argument("--gpu", type=int, action="append", default=[])

group = parser.add_argument_group("Validation Options")
group.add_argument("--validate", action="store_true", default=False)
for mode in MODES:
    group.add_argument(f"--val-{mode}-path")

group = parser.add_argument_group("Word Embeddings Options")
embeds.add_embed_arguments(group)


def create_dataloader(args, vocabs=None, val=False):
    argvalpfx = "val_" if val else ""
    paths = [getattr(args, f"{argvalpfx}{mode}_path") for mode in MODES]
    if vocabs is None:
        vocabs = [getattr(args, f"{mode}_vocab") for mode in MODES]
        vocabs = [utils.load_pkl(v) if v is not None else None for v in vocabs]
    dset = dataset.TextSequenceDataset(
        paths=paths,
        feats=["string", "tensor"],
        vocabs=vocabs,
        vocab_limit=args.vocab_limit,
        pad_eos=args.eos,
        pad_bos=args.bos,
        unk=args.unk,
    )
    if vocabs is None:
        vocabs = dset.vocabs
    collator = dataset.TextSequenceBatchCollator(
        pad_idxs=[len(v) for v in vocabs]
    )
    return td.DataLoader(
        dataset=dset,
        batch_size=args.batch_size,
        shuffle=False if val else args.shuffle,
        num_workers=args.data_workers,
        collate_fn=collator,
        pin_memory=args.pin_memory
    )


def prepare_model(args, vocabs, resume_from=None):
    if resume_from is None:
        resume_from = dict()

    model_path = args.model_path
    if resume_from.get("model_args") is not None:
        temp_path = tempfile.mkstemp()[1]
        utils.dump_yaml(resume_from["model_args"], temp_path)

    torchmodels.register_packages(models)
    mdl_cls = torchmodels.create_model_cls(models.jlu, model_path)
    mdl = mdl_cls(
        hidden_dim=args.hidden_dim,
        word_dim=args.word_dim,
        num_words=len(vocabs[0]),
        num_slots=len(vocabs[1]),
        num_intents=len(vocabs[2])
    )
    mdl.reset_parameters()
    if resume_from.get("model") is not None:
        mdl.load_state_dict(resume_from["model"])
    else:
        embeds.load_embeddings(args, vocabs[0], mdl.embeddings())
    return mdl


def get_optimizer_cls(args):
    kwargs = dict()
    if args.learning_rate is not None:
        kwargs["lr"] = args.learning_rate
    if args.weight_decay is not None:
        kwargs["weight_decay"] = args.weight_decay
    return utils.map_val(args.optimizer, {
        "sgd": lambda p: op.SGD(p, **kwargs),
        "adam": lambda p: op.Adam(p, **kwargs),
        "adamax": lambda p: op.Adamax(p, **kwargs),
        "adagrad": lambda p: op.Adagrad(p, **kwargs),
        "adadelta": lambda p: op.Adadelta(p, **kwargs)
    }, "optimizer")


def normalize(x):
    return x / sum(x)


def randidx(x, size):
    """x is either integer or array-like probability distribution"""
    if isinstance(x, int):
        return torch.randint(0, x, size)
    else:
        return np.random.choice(np.arange(len(x)), p=x, size=size)


class Validator(inference.LoggableLossInferencer,
                inference.PredictingLossInferencer):

    def __init__(self, *args, **kwargs):
        super(Validator, self).__init__(
            *args,
            name="valid",
            tensorboard=False,
            persistant_steps=False,
            **kwargs
        )

    def on_run_started(self, dataloader):
        super(Validator, self).on_run_started(dataloader)
        self.labels_gold, self.intents_gold = list(), list()

    def on_batch_started(self, batch):
        super(Validator, self).on_batch_started(batch)
        self.model.train(False)

    def on_loss_calculated(self, batch, data, model_outputs, losses, stats):
        super(Validator, self)\
            .on_loss_calculated(batch, data, model_outputs, losses, stats)
        # dataset might produce bos/eos-padded strings
        labels_gold = [self.trim(x[1]) for x in batch["string"]]
        intents_gold = [self.trim(x[2]) for x in batch["string"]]
        self.labels_gold.extend(labels_gold)
        self.intents_gold.extend(intents_gold)

    def on_run_finished(self, stats):
        preds = super(Validator, self).on_run_finished(stats)
        assert preds is not None, "polymorphism gone wrong?"
        lpreds, ipreds = [x[0][0] for x in preds], [x[1][0] for x in preds]
        res = evaluate.evaluate(
            gold_labels=self.labels_gold,
            gold_intents=self.intents_gold,
            pred_labels=lpreds,
            pred_intents=ipreds
        )
        tasks = ["slot-labeling", "intent-classification"]
        stats = {
            f"val-{measure}-{mode}": v
            for mode, task in zip(MODES[1:], tasks)
            for measure, v in res[task]["overall"].items()
        }
        stats["val-acc-sent"] = res["sentence-understanding"]
        msg = utils.join_dict(
            {k: f"{v:.4f}" for k, v in stats.items()},
            item_dlm=", ", kvp_dlm="="
        )
        self.log(msg, tag="eval")
        return stats


class Trainer(inference.LoggableLossInferencer):

    def __init__(self, *args, epochs=10, optimizer_cls=op.Adam, model_path=None,
                 save_period=None, samples=None, validator=None,
                 early_stop=False, early_stop_patience=5,
                 early_stop_criterion="val-acc-sent", **kwargs):
        super(Trainer, self).__init__(
            *args,
            name="train",
            persistant_steps=True,
            **kwargs
        )
        self.epochs = epochs
        self.optimizer_cls = optimizer_cls
        self.show_samples = samples is not None
        self.num_samples = samples
        self.should_validate = validator is not None
        self.validator = validator
        self.should_save_periodically = save_period is not None
        self.save_period = save_period
        self.model_path = model_path
        self.early_stop = early_stop
        self.early_stop_patience = early_stop_patience
        self.early_stop_criterion = early_stop_criterion

        self.global_step = 0
        self.eidx = 0
        self.optimizer = optimizer_cls(self.trainable_params())
        self.early_stop_best = {
            "crit": None,
            "eidx": -1,
            "sd": None,
            "stats": None
        }

    def trainable_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                yield p

    def save_snapshot(self, state_dict, tag=None):
        if tag is None:
            tag = ""
        else:
            tag = f"-{tag}"
        eidx = state_dict["eidx"]
        path = os.path.join(self.save_dir, f"checkpoint-e{eidx:02d}{tag}")
        torch.save(state_dict, path)
        logging.info(f"checkpoint saved to '{path}'.")

    def snapshot(self, stats=None):
        exp_state_dict = {
            "eidx": self.eidx,
            "model": {
                k: v.detach().cpu()
                for k, v in self.module.state_dict().items()
            },
            "global_step": self.global_step,
            "optimizer": self.optimizer.state_dict(),
        }
        if self.model_path is not None:
            exp_state_dict["model_args"] = utils.load_yaml(self.model_path)
        if stats is not None:
            exp_state_dict["stats"] = stats
        return exp_state_dict

    def load_snapshot(self, state_dict):
        if self.optimizer is not None and "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        if "global_step" in state_dict:
            self.global_step = state_dict["global_step"]
        if "eidx" in state_dict:
            self.eidx = state_dict["eidx"]

    def should_stop(self, eidx, stats):
        if not self.early_stop:
            return False
        assert self.early_stop_criterion in stats, \
            f"early stop criterion not found in training stats: " \
            f"{self.early_stop_criterion} not in {stats.keys()}"
        crit = stats.get(self.early_stop_criterion)
        if self.early_stop_best["crit"] is None or crit > self.early_stop_best["crit"]:
            self.early_stop_best["crit"] = crit
            self.early_stop_best["sd"] = self.snapshot(stats)
        return self.early_stop_best["sd"]["eidx"] is not None and \
               eidx >= self.early_stop_best["sd"]["eidx"] + \
                       self.early_stop_patience

    def report_early_stop(self, eidx):
        stats_str = {k: f"{v:.4f}" for k, v in
                     self.early_stop_best["sd"]["stats"].items()}
        logging.info(f"early stopping at {eidx} epoch as criterion "
                     f"({self.early_stop_criterion}) remains unchallenged "
                     f"for {self.early_stop_patience} epochs.")
        logging.info(f"best stats so far:")
        logging.info(f"[{eidx - self.early_stop_patience}] "
                     f"{utils.join_dict(stats_str, ', ', '=')}")

    def on_batch_started(self, batch):
        super(Trainer, self).on_batch_started(batch)
        self.model.train(True)
        self.optimizer.zero_grad()

    def on_loss_calculated(self, batch, data, model_outputs, losses, stats):
        ret = super(Trainer, self)\
            .on_loss_calculated(batch, data, model_outputs, losses, stats)
        loss = losses["loss-total"]
        loss.backward()
        self.optimizer.step()
        return ret

    def train(self, dataloader, val_dataloader=None):
        if self.should_validate:
            assert val_dataloader is not None, \
                "must provide validation data if i need to validate"
        self.optimizer = self.optimizer_cls(self.trainable_params())
        self.progress_global = utils.tqdm(
            total=self.epochs,
            desc=f"training {self.epochs} epochs",
            disable=not self.show_progress
        )
        self.progress_global.update(self.eidx)
        for self.eidx in range(self.eidx + 1, self.epochs + 1):
            self.progress_global.update(1)
            self.progress_global.set_description(f"training e{self.eidx:02d}")
            stats = self.inference(dataloader)

            if self.should_validate:
                with torch.no_grad():
                    valstats = self.validator.inference(val_dataloader)
                stats.update(valstats)

            if self.should_save_periodically \
                and self.eidx % self.save_period == 0:
                self.save_snapshot(self.snapshot())

            if self.should_stop(self.eidx, stats):
                self.save_snapshot(
                    state_dict=self.early_stop_best["sd"],
                    tag=f"best-{self.early_stop_best['crit']:.4f}"
                )
                self.report_early_stop(self.eidx)
                break
        self.progress_global.close()


def report_model(trainer):
    params = sum(np.prod(p.size()) for p in trainer.trainable_params())
    logging.info(f"Number of parameters: {params:,}")
    logging.info(f"{trainer.module}")


def train(args):
    devices = utils.get_devices(args.gpu)
    if args.seed is not None:
        utils.manual_seed(args.seed)

    logging.info("Loading data...")
    dataloader = create_dataloader(args)
    vocabs = dataloader.dataset.vocabs
    if args.validate:
        val_dataloader = create_dataloader(args, vocabs, True)
    else:
        val_dataloader = None
    fnames = [f"{mode}.vocab" for mode in MODES]
    for vocab, fname in zip(vocabs, fnames):
        utils.save_pkl(vocab, os.path.join(args.save_dir, fname))

    logging.info("Initializing training environment...")
    resume_from = dict()
    if args.resume_from is not None:
        resume_from = torch.load(args.resume_from)
    mdl = prepare_model(args, vocabs, resume_from)
    mdl = utils.to_device(mdl, devices)
    optimizer_cls = get_optimizer_cls(args)
    validator = None
    if args.validate:
        validator = Validator(
            model=mdl,
            device=devices[0],
            vocabs=vocabs,
            bos=args.bos,
            eos=args.eos,
            unk=args.unk,
            alpha=args.loss_alpha,
            beta=args.loss_beta,
            progress=args.show_progress,
            batch_stats=args.log_stats
        )
    trainer = Trainer(
        model=mdl,
        model_path=args.model_path,
        alpha=args.loss_alpha,
        beta=args.loss_beta,
        device=devices[0],
        vocabs=vocabs,
        epochs=args.epochs,
        save_dir=args.save_dir,
        save_period=args.save_period,
        optimizer_cls=optimizer_cls,
        samples=args.samples,
        tensorboard=args.tensorboard,
        progress=args.show_progress,
        validator=validator,
        batch_stats=args.log_stats,
        early_stop=args.early_stop,
        early_stop_criterion=args.early_stop_criterion,
        early_stop_patience=args.early_stop_patience
    )
    trainer.load_snapshot(resume_from)
    report_model(trainer)

    logging.info("Commencing training joint-lu...")
    trainer.train(dataloader, val_dataloader)

    logging.info("Done!")


if __name__ == "__main__":
    train(utils.initialize_script(parser))