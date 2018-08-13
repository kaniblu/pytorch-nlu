import os
import logging
import argparse
import importlib
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.optim as op
import torch.utils.data as td

import utils
import model
import model.utils
import dataset
import predict
import evaluate
from . import embeds

MODES = model.MODES

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument("--debug", action="store_true", default=False)

group = parser.add_argument_group("Logging Options")
utils.add_logging_arguments(group, "train")
group.add_argument("--show-progress", action="store_true", default=False)

group = parser.add_argument_group("Model Options")
model.add_arguments(group)

group = parser.add_argument_group("Data Options")
for mode in MODES:
    group.add_argument(f"--{mode}-path", type=str, required=True)
    group.add_argument(f"--{mode}-vocab", type=str, default=None)
group.add_argument("--vocab-limit", type=int, default=None)
group.add_argument("--data-workers", type=int, default=8)
group.add_argument("--pin-memory", action="store_true", default=False)
group.add_argument("--shuffle", action="store_true", default=False)
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
group.add_argument("--early-stop", action="store_true", default=False)
group.add_argument("--early-stop-patience", type=int, default=5)
group.add_argument("--early-stop-save", action="store_true", default=False)
group.add_argument("--early-stop-criteria", default="acc-label")
group.add_argument("--learning-rate", type=float, default=None)
group.add_argument("--weight-decay", type=float, default=None)
group.add_argument("--samples", type=int, default=1)
group.add_argument("--tensorboard", action="store_true", default=False)
group.add_argument("--gpu", type=int, action="append", default=[])

group = parser.add_argument_group("Validation Options")
group.add_argument("--validate", action="store_true", default=False)
for mode in MODES:
    group.add_argument(f"--val-{mode}-path")

group = parser.add_argument_group("Word Embeddings Options")
for mode in MODES:
    embeds.add_embed_arguments(group, mode)


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
        shuffle=False if val else args.shuffle ,
        num_workers=args.data_workers,
        collate_fn=collator,
        pin_memory=args.pin_memory
    )


def prepare_model(args, vocabs):
    mdl = model.create_model(args, vocabs)
    mdl.reset_parameters()
    for mode, vocab, emb in zip(MODES, vocabs, mdl.embeddings()):
        embeds.load_embeddings(
            args=utils.filter_namespace_prefix(args, mode),
            vocab=vocab,
            modules=[emb]
        )
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


class Trainer(object):
    def __init__(self, debug, model, device, vocabs, epochs, save_dir,
                 save_period, optimizer_cls=op.Adam, enable_tensorboard=False,
                 show_progress=True, samples=None, tensor_key="tensor",
                 predictor=None, validate=1,
                 earlystop=False, patience=5, earlystop_save=False,
                 earlystop_criteria="val-acc-sent"):
        self.debug = debug
        self.model = model
        self.device = device
        self.epochs = epochs
        self.vocabs = vocabs
        self.save_dir = save_dir
        self.save_period = save_period
        self.optimizer_cls = optimizer_cls
        self.tensor_key = tensor_key
        self.enable_tensorboard = enable_tensorboard
        self.cross_entropies = [nn.CrossEntropyLoss(ignore_index=len(vocab))
                                for vocab in vocabs]
        self.show_progress = show_progress
        self.samples = samples
        self.unk = "<unk>"
        self.perform_validation = validate
        self.predictor = predictor
        self.earlystop = earlystop,
        self.earlystop_patience = patience
        self.earlystop_save = earlystop_save
        self.earlystop_criteria = earlystop_criteria
        self.earlystop_max = None
        self.earlystop_max_eidx = None
        self.earlystop_max_sd = None
        self.earlystop_max_stats = None

        if self.enable_tensorboard:
            self.tensorboard = importlib.import_module("tensorboardX")
            self.writer = self.tensorboard.SummaryWriter(log_dir=self.save_dir)

    @property
    def module(self):
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        else:
            return self.model

    def trainable_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                yield p

    def prepare_batch(self, batch):
        data = batch[self.tensor_key]
        data = [(x.to(self.device), lens.to(self.device)) for x, lens in data]
        (w, w_lens), (l, l_lens), (i, i_lens) = data
        batch_size = w.size(0)
        if self.debug:
            assert (w_lens == l_lens).sum().item() == batch_size
            assert (i_lens == 3).sum().item() == batch_size
        return batch_size, (w, l, i[:, 1], w_lens)

    def snapshot(self, eidx, state_dict=None):
        if state_dict is None:
            state_dict = self.module.state_dict()
        path = os.path.join(self.save_dir, f"checkpoint-e{eidx:02d}")
        torch.save(state_dict, path)
        logging.info(f"checkpoint saved to '{path}'.")

    def report_stats(self, stats):
        if self.enable_tensorboard:
            for k, v in stats.items():
                self.writer.add_scalar(k.replace("-", "/"), v, self.global_step)
        stats_str = {k: f"{v:.4f}" for k, v in stats.items()}
        desc = utils.join_dict(stats_str, ", ", "=")
        return desc

    def report_samples(self, golds, logits=None, vocab=None,
                       name="", lens=None, idx=None):
        def to_sent(vec):
            return " ".join(vocab.i2f.get(w, self.unk) for w in vec)
        if self.samples is None:
            return
        if idx is None:
            idx = torch.randperm(len(golds))[:self.samples]
        golds = [golds[i.item()] for i in idx]
        if logits is None:
            for i, sent in enumerate(golds):
                logging.info(f"{name.capitalize()} Sample #{i + 1:02d}:")
                logging.info(f"Target:    {sent}")
        else:
            preds = logits.max(2)[1]
            if lens is None:
                lens = torch.full((len(logits), ), preds.size(1)).long()
            preds, lens = preds[idx].cpu().tolist(), lens[idx].cpu().tolist()
            preds = [to_sent(pred[:l]) for pred, l in zip(preds, lens)]
            for i, (sent, pred) in enumerate(zip(golds, preds)):
                logging.info(f"{name.capitalize()} Sample #{i + 1:02d}:")
                logging.info(f"Target:    {sent}")
                logging.info(f"Predicted: {pred}")

    def report_all_samples(self, lgts, igts, wgld, lgld, igld, lens):
        if self.samples is None:
            return
        igts = igts.unsqueeze(1)
        idx = torch.randperm(len(lgts))[:self.samples]
        self.report_samples(wgld, name="word", idx=idx)
        self.report_samples(lgld, lgts, self.vocabs[1], "label", lens, idx)
        self.report_samples(igld, igts, self.vocabs[2], "intent", idx=idx)

    def calculate_celoss(self, ce, logits, targets):
        logit_size = logits.size(-1)
        logits = logits.view(-1, logit_size)
        targets = targets.view(-1)
        return ce(logits, targets)

    def calculate_losses(self, logits_lst, gold_lst):
        return {
            f"loss-{mode}": self.calculate_celoss(ce, logits, gold)
            for mode, ce, logits, gold in
            zip(["label", "intent"], self.cross_entropies, logits_lst, gold_lst)
        }

    def calculate_acc(self, logits, targets, ignore_index):
        preds = logits.max(-1)[1]
        mask = (targets != ignore_index).float()
        correct = ((preds == targets).float() * mask).sum().item()
        total = mask.sum().item()
        return correct / total

    def calculate_accuracies(self, logits_lst, gold_lst):
        return {
            f"acc-{mode}": self.calculate_acc(logits, golds, len(vocab))
            for mode, logits, golds, vocab in
            zip(MODES[1:], logits_lst, gold_lst, self.vocabs)
        }

    def trim_beos(self, sent):
        return " ".join(sent.split()[1:-1])

    def validate(self, eidx, dataloader):
        if self.predictor is None:
            return
        with torch.no_grad():
            (lpreds, ipreds), _ = self.predictor.predict(dataloader)
        lpreds = [self.trim_beos(l) for l in lpreds]
        lgolds, igolds = [], []
        for batch in dataloader:
            _, lb, ib = list(zip(*batch["string"]))
            lgolds.extend([self.trim_beos(l) for l in lb])
            igolds.extend([self.trim_beos(i) for i in ib])
        res = evaluate.evaluate(lgolds, igolds, lpreds, ipreds)
        tasks = ["slot-labeling", "intent-classification"]
        valstats = {
            f"val-{measure}-{mode}": v
            for mode, task in zip(MODES[1:], tasks)
            for measure, v in res[task]["overall"].items()
        }
        valstats["val-acc-sent"] = res["sentence-understanding"]
        logging.info(f"[{eidx}] {self.report_stats(valstats)}")
        return valstats

    def should_stop(self, eidx, stats):
        if not self.earlystop:
            return False
        assert self.earlystop_criteria in stats, \
            f"early stop criteria not found in training stats: " \
            f"{self.earlystop_criteria} not in {stats.keys()}"
        crit = stats.get(self.earlystop_criteria)
        if self.earlystop_max is None or crit > self.earlystop_max:
            self.earlystop_max = crit
            self.earlystop_max_eidx = eidx
            self.earlystop_sd = {
                k: v.detach() for k, v in self.module.state_dict().items()
            }
            self.earlystop_max_stats = stats
        return self.earlystop_max_eidx is not None and \
               eidx >= self.earlystop_max_eidx + self.earlystop_patience

    def report_early_stop(self, eidx):
        stats_str = {k: f"{v:.4f}" for k, v in self.earlystop_max_stats.items()}
        logging.info(f"early stopping at {eidx} epoch as criteria "
                     f"({self.earlystop_criteria}) remains unchallenged "
                     f"for {self.earlystop_patience} epochs.")
        logging.info(f"best stats so far:")
        logging.info(f"[{eidx - self.earlystop_patience}] "
                     f"{utils.join_dict(stats_str, ', ', '=')}")

    def train(self, dataloader, val_dataloader=None):
        self.global_step = 0
        optimizer = self.optimizer_cls(list(self.trainable_params()))
        progress_global = utils.tqdm(
            total=self.epochs,
            desc=f"training {self.epochs} epochs",
            disable=not self.show_progress
        )

        for eidx in range(1, self.epochs + 1):
            self.local_step = 0
            stats_cum = collections.defaultdict(float)
            progress_global.update(1)
            progress_local = utils.tqdm(
                total=len(dataloader.dataset),
                desc=f"training an epoch",
                disable=not self.show_progress
            )
            for batch in dataloader:
                self.model.train(True)
                optimizer.zero_grad()
                batch_size, (w, l, i, lens) = self.prepare_batch(batch)
                self.global_step += batch_size
                self.local_step += batch_size
                progress_local.update(batch_size)
                ret = self.model(w, l, lens)
                label_logits, intent_logits = ret.get("pass")
                losses = self.calculate_losses(
                    logits_lst=[label_logits, intent_logits],
                    gold_lst=[l, i]
                )
                loss = sum(losses.values())
                loss.backward()
                if self.debug:
                    for p in self.model.parameters():
                        if p.grad is None:
                            continue
                        if (p.grad != p.grad).float().sum().item() > 0:
                            logging.error(f"[epoch: {eidx}] NaN detected in "
                                          f"gradients of parameter "
                                          f"(size: {p.size()})")
                optimizer.step()

                stats = losses
                stats.update(self.calculate_accuracies(
                    logits_lst=[label_logits, intent_logits],
                    gold_lst=[l, i]
                ))
                for k, v in stats.items():
                    stats_cum[k] += v * batch_size

                items_lst = list(zip(*batch.get("string")))
                self.report_all_samples(
                    lgts=label_logits,
                    igts=intent_logits,
                    wgld=items_lst[0],
                    lgld=items_lst[1],
                    igld=items_lst[2],
                    lens=lens
                )

            progress_local.close()
            stats_cum = {k: v / self.local_step for k, v in stats_cum.items()}
            logging.info(f"[{eidx}] {self.report_stats(stats_cum)}")

            if self.perform_validation:
                stats_cum.update(self.validate(eidx, val_dataloader))

            if self.save_period is not None and \
                    eidx % self.save_period == 0:
                self.snapshot(eidx)

            if self.should_stop(eidx, stats_cum):
                if self.earlystop_save:
                    self.snapshot(
                        eidx=self.earlystop_max_eidx,
                        state_dict=self.earlystop_max_sd
                    )
                self.report_early_stop(eidx)
                break
        progress_global.close()


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
    mdl = prepare_model(args, vocabs)
    mdl = utils.to_device(mdl, devices)
    optimizer_cls = get_optimizer_cls(args)
    if args.validate:
        predictor = predict.Predictor(
            model=mdl,
            device=devices[0],
            sent_vocab=vocabs[0],
            label_vocab=vocabs[1],
            intent_vocab=vocabs[2],
            bos=args.bos,
            eos=args.eos,
            unk=args.unk,
            tensor_key="tensor"
        )
    else:
        predictor = None
    trainer = Trainer(
        debug=args.debug,
        model=mdl,
        device=devices[0],
        vocabs=vocabs,
        epochs=args.epochs,
        save_dir=args.save_dir,
        save_period=args.save_period,
        optimizer_cls=optimizer_cls,
        tensor_key="tensor",
        samples=args.samples,
        enable_tensorboard=args.tensorboard,
        show_progress=args.show_progress,
        predictor=predictor,
        validate=args.validate,
        earlystop=args.early_stop,
        earlystop_criteria=args.early_stop_criteria,
        earlystop_save=args.early_stop_save,
        patience=args.early_stop_patience
    )
    report_model(trainer)

    logging.info("Commencing training joint-lu...")
    trainer.train(dataloader, val_dataloader)

    logging.info("Done!")


if __name__ == "__main__":
    train(utils.initialize_script(parser))