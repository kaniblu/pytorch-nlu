

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


def evaluate_intents(golds, preds):
    vocab = set(golds) | set(preds)
    measures = (
        metrics.precision_score,
        metrics.recall_score,
        metrics.f1_score
    )
    ret = {
        "overall": {
            "acc": metrics.accuracy_score(golds, preds),
            "prec": metrics.precision_score(golds, preds, average="micro"),
            "rec": metrics.recall_score(golds, preds, average="micro"),
            "f1": metrics.f1_score(golds, preds, average="micro"),
        },
        "intents": {
            intent: {
                name: measure(
                    y_true=replace(golds, intent, 1, 0),
                    y_pred=replace(preds, intent, 1, 0),
                    average="binary"
                ) for name, measure in zip(("prec", "rec", "f1"), measures)
            } for intent in vocab
        }
    }
    return cast_float(ret)


def evaluate(gold_labels, gold_intents, pred_labels, pred_intents):
    return {
        "intent-classification": evaluate_intents(gold_intents, pred_intents),
        "slot-labeling": slots.evaluate(gold_labels, pred_labels),
        "sentence-understanding": float(metrics.accuracy_score(
            y_true=list(map("/".join, zip(gold_labels, gold_intents))),
            y_pred=list(map("/".join, zip(pred_labels, pred_intents)))
        ))
    }