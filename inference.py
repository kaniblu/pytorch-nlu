import logging
import importlib
import collections

import torch.nn as nn
import torch.nn.functional as F

import utils

MODES = ["word", "label", "intent"]


class Inferencer(object):

    def __init__(self, model, device, vocabs, alpha=1.0, beta=1.0):
        """alpha: weight coefficient for intent loss"""
        self.debug = False
        self.model = model
        self.device = device
        self.vocabs = vocabs
        self.alpha = alpha
        self.beta = beta
        self.cross_entropies = [nn.CrossEntropyLoss(ignore_index=len(vocab))
                                for vocab in vocabs]

    @property
    def module(self):
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        else:
            return self.model

    def prepare_batch(self, batch):
        data = batch["tensor"]
        data = [(x.to(self.device), lens.to(self.device)) for x, lens in
                data]
        (w, w_lens), (l, l_lens), (i, i_lens) = data
        batch_size = w.size(0)
        if self.debug:
            assert (w_lens == l_lens).sum().item() == batch_size
            assert (i_lens == 3).sum().item() == batch_size
        return batch_size, (w, l, i[:, 1], w_lens)

    def calculate_acc(self, logits, targets, ignore_index):
        preds = logits.max(-1)[1]
        mask = (targets != ignore_index).float()
        correct = ((preds == targets).float() * mask).sum().item()
        total = mask.sum().item()
        if total == 0.0:
            return 0.0
        return correct / total

    def calculate_accuracies(self, logits_lst, gold_lst):
        return {
            f"acc-{mode}": self.calculate_acc(logits, golds, len(vocab))
            for mode, logits, golds, vocab in
            zip(MODES[1:], logits_lst, gold_lst, self.vocabs)
        }

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

    def on_run_started(self, dataloader):
        logging.info(f"{self.__class__.__name__} run started")

    def on_run_finished(self, stats):
        logging.info(f"{self.__class__.__name__} run ended")
        return stats

    def on_batch_started(self, batch):
        pass

    def on_batch_finished(self, batch, model_outputs, losses, stats):
        pass

    def inference(self, dataloader):
        self.on_run_started(dataloader)

        self.step = 0
        stats_cum = collections.defaultdict(float)

        for batch in dataloader:
            self.on_batch_started(batch)

            batch_size, (w, l, i, lens) = self.prepare_batch(batch)
            self.step += batch_size

            label_logits, intent_logits = self.model(w, l, lens)
            losses = self.calculate_losses(
                logits_lst=[label_logits, intent_logits],
                gold_lst=[l, i]
            )
            losses["loss-total"] = self.beta * losses["loss-label"] + \
                                   self.alpha * losses["loss-intent"]
            stats = {k: v.cpu().item() for k, v in losses.items()}
            stats.update(self.calculate_accuracies(
                logits_lst=[label_logits, intent_logits],
                gold_lst=[l, i]
            ))
            for k, v in stats.items():
                stats_cum[k] += v * batch_size

            self.on_batch_finished(
                batch=batch,
                model_outputs=(label_logits, intent_logits),
                losses=losses,
                stats=stats
            )
        stats_cum = {k: v / self.step for k, v in stats_cum.items()}
        return self.on_run_finished(stats_cum)


class LoggableInferencer(Inferencer):

    def __init__(self, *args, name, save_dir=None, persistant_steps=False,
                 progress=False, tensorboard=False, batch_stats=False, **kwargs):
        super(LoggableInferencer, self).__init__(*args, **kwargs)
        self.name = name
        self.save_dir = save_dir
        self.show_progress = progress
        self.enable_tensorboard = tensorboard
        self.should_report_batch_stats = batch_stats
        self.persistant_steps = persistant_steps
        self.global_step = 0

        if self.enable_tensorboard:
            tbx = importlib.import_module("tensorboardX")
            self.writer = tbx.SummaryWriter(log_dir=self.save_dir)

    def log(self, msg, tag=None, level=logging.INFO, step=None):
        if step is None:
            if not self.persistant_steps:
                step = self.step
            else:
                step = self.global_step
        header = f"({self.name}|s{step})"
        if tag is None:
            msg = f"{header} {msg}"
        else:
            msg = f"{header} {tag}: {msg}"
        logging.log(level=level, msg=msg)

    def report_stats(self, stats, step=None, log=True):
        if step is None:
            if not self.persistant_steps:
                step = self.step
            else:
                step = self.global_step
        if self.enable_tensorboard:
            for k, v in stats.items():
                self.writer.add_scalar(k.replace("-", "/"), v, step)
        if log:
            stats_str = {k: f"{v:.4f}" for k, v in stats.items()}
            desc = utils.join_dict(stats_str, ", ", "=")
            self.log(desc)

    def on_run_started(self, dataloader):
        super(LoggableInferencer, self).on_run_started(dataloader)
        self.progress = utils.tqdm(
            total=len(dataloader.dataset),
            desc=f"running {self.name}",
            disable=not self.show_progress
        )

    def on_batch_started(self, batch):
        super(LoggableInferencer, self).on_batch_started(batch)
        batch_size = len(batch["string"])
        self.global_step += batch_size
        self.progress.update(batch_size)

    def on_batch_finished(self, batch, model_outputs, losses, stats):
        super(LoggableInferencer, self)\
            .on_batch_finished(batch, model_outputs, losses, stats)
        self.report_stats(stats, log=self.should_report_batch_stats)

    def on_run_finished(self, stats):
        ret = super(LoggableInferencer, self).on_run_finished(stats)
        self.progress.close()
        self.report_stats(stats)
        return ret


class Predictor(Inferencer):

    def __init__(self, *args, unk="<unk>", bos="<bos>", eos="<eos>", **kwargs):
        super(Predictor, self).__init__(*args, **kwargs)
        self.bos = bos
        self.eos = eos
        self.unk = unk

    def to_sent(self, idx, vocab):
        return " ".join(vocab.i2f.get(w, self.unk) for w in idx)

    def trim(self, sent):
        tokens = sent.split()
        if tokens[0] == self.bos:
            tokens = tokens[1:]
        if tokens[-1] == self.eos:
            tokens = tokens[:-1]
        return " ".join(tokens)

    def on_run_started(self, dataloader):
        super(Predictor, self).on_run_started(dataloader)
        self.labels, self.intents = list(), list()
        self.lprobs, self.iprobs = list(), list()

    def on_batch_finished(self, batch, model_outputs, losses, stats):
        super(Predictor, self)\
            .on_batch_finished(batch, model_outputs, losses, stats)

        lgts, igts = model_outputs
        sprobs = F.softmax(lgts, 2)
        iprobs = F.softmax(igts, 1)
        sprobs, slots = sprobs.max(2)
        sprobs = sprobs.prod(1)
        iprobs, intents = iprobs.max(1)
        lens = batch["tensor"][0][1]

        slots, intents, sprobs, iprobs, lens = \
            [x.cpu().tolist() for x in [slots, intents, sprobs, iprobs, lens]]
        slots = [self.to_sent(s[:l], self.vocabs[1])
                 for s, l in zip(slots, lens)]
        slots = [self.trim(s) for s in slots]
        intents = [self.to_sent([i], self.vocabs[2]) for i in intents]
        self.labels.extend(slots)
        self.lprobs.extend(sprobs)
        self.intents.extend(intents)
        self.iprobs.extend(iprobs)

        return [((s, sp), (i, ip)) for s, i, sp, ip in
                zip(slots, intents, sprobs, iprobs)]

    def on_run_finished(self, stats):
        super(Predictor, self).on_run_finished(stats)
        return [((s, sp), (i, ip)) for s, i, sp, ip in
                zip(self.labels, self.intents, self.lprobs, self.iprobs)]