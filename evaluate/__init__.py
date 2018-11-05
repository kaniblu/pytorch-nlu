from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

from . import slots


def replace(items, x, y, z):
    return [y if item == x else z for item in items]


def cast_float(d):
    if isinstance(d, dict):
        return {k: cast_float(v) for k, v in d.items()}
    else:
        return float(d)


def measure_count(y_true, y_pred, **kwargs):
    return len([x for x in y_true if x == 1])


def evaluate_intents(golds, preds, detailed=False):
    vocab = set(golds) | set(preds)
    measures = (
        metrics.precision_score,
        metrics.recall_score,
        metrics.f1_score,
        measure_count
    )
    ret = {
        "overall": {
            "acc": metrics.accuracy_score(golds, preds),
            "prec": metrics.precision_score(golds, preds, average="micro"),
            "rec": metrics.recall_score(golds, preds, average="micro"),
            "f1": metrics.f1_score(golds, preds, average="micro"),
        }
    }
    if detailed:
        ret.update({
            "intents": {
                intent: {
                    name: measure(
                        y_true=replace(golds, intent, 1, 0),
                        y_pred=replace(preds, intent, 1, 0),
                        average="binary"
                    ) for name, measure in
                    zip(("prec", "rec", "f1", "counts"), measures)
                } for intent in vocab
            }
        })
    return cast_float(ret)

import logging
def evaluate(gold_labels, gold_intents, pred_labels, pred_intents, detailed=False):
    ret = dict()
    ret["intent-classification"] = evaluate_intents(gold_intents, pred_intents, detailed)
    ret["slot-labeling"] = slots.ConllEvaluator(detailed).evaluate(
        golds=[l.split() for l in gold_labels],
        preds=[l.split() for l in pred_labels]
    )
    ret["sentence-understanding"] = float(metrics.accuracy_score(
        y_true=list(map("/".join, zip(gold_labels, gold_intents))),
        y_pred=list(map("/".join, zip(pred_labels, pred_intents)))
    ))
    return ret
