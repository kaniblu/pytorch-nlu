import os
import re
import sys
import stat
import subprocess


class ConllEvalParser(object):
    MEASURE = {
        "FB1": "f1",
        "precision": "prec",
        "recall": "rec",
        "accuracy": "acc"
    }
    def __init__(self):
        self.pat_kvp = re.compile(r"(\w+):\s+([0-9.]+)%?")
        self.pat_line = re.compile(r"^\s*([\w.]*):(.*)(\d+)\s*$")

    def parse_kvp(self, text):
        match = self.pat_kvp.search(text)
        assert match is not None
        return match.group(1), float(match.group(2)) / 100

    def parse_line(self, line):
        match = self.pat_line.search(line)
        assert match is not None
        slot = match.group(1)
        kvps_txt = match.group(2)
        count = int(match.group(3))
        measure = dict(self.parse_kvp(t) for t in kvps_txt.split(";"))
        measure = {self.MEASURE[k]: v for k, v in measure.items()}
        if slot == "":
            slot = "O"
        return slot, dict(
            count=count,
            results=measure
        )

    def parse(self, text):
        lines = text.splitlines()
        ret = dict(
            overall={
                self.MEASURE[k]: v
                for k, v in dict(self.parse_kvp(t)
                                 for t in lines[1].split(";")).items()
            }
        )
        ret["slots"] = dict(self.parse_line(line) for line in lines[2:])
        return ret


def evaluate_conlleval(golds, preds):
    """
    Evaluates slot accuracies using conlleval.pl script
    conlleval.pl has been converted to binary for portability
    :param golds: list of space-separated strings
    :param preds: list of space-separated strings
    :return: dict
    """
    bindir = os.path.dirname(os.path.realpath(__file__))
    binpath = os.path.join(bindir, "conlleval")
    os.chmod(binpath, os.stat(binpath).st_mode | stat.S_IEXEC)
    path = [bindir] + sys.path
    p = subprocess.Popen(
        args=["conlleval"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            "PATH": ":".join(path)
        }
    )
    for gold, pred in zip(golds, preds):
        for s1, s2 in zip(gold.split(), pred.split()):
            p.stdin.write(f"{s1} {s2}\n".encode())
        p.stdin.write(b"\n")
    p.stdin.close()
    text = p.stdout.read()
    parser = ConllEvalParser()
    return parser.parse(text.decode())


class ConllEvaluator(object):

    START_TAGS = {
        ("B", "B"),
        ("I", "B"),
        ("O", "B"),
        ("O", "I"),
        ("E", "E"),
        ("E", "I"),
        ("O", "E"),
        ("O", "I")
    }
    END_TAGS = {
        ("B", "B"),
        ("B", "O"),
        ("I", "B"),
        ("I", "O"),
        ("E", "E"),
        ("E", "I"),
        ("E", "O"),
        ("I", "O")
    }

    def __init__(self, detailed=False):
        self.detailed = detailed

    def is_chunk_start(self, ptag, tag, ptag_type, tag_type):
        return (ptag, tag) in self.START_TAGS or \
               (tag != "O" and tag != "." and ptag_type != tag_type)

    def is_chunk_end(self, ptag, tag, ptag_type, tag_type):
        return (ptag, tag) in self.END_TAGS or \
               ptag != 'O' and ptag != '.' and ptag_type != tag_type

    @staticmethod
    def split_tag(tag):
        s = tag.split('-')
        if len(s) > 2 or len(s) == 0:
            raise ValueError('tag format wrong. it must be B-xxx.xxx')
        if len(s) == 1:
            tag = s[0]
            tag_type = ""
        else:
            tag = s[0]
            tag_type = s[1]
        return tag, tag_type

    @staticmethod
    def form_stats(num_correct, num_pred, num_gold):
        rec, prec, f1 = 0.0, 0.0, 0.0
        if num_pred > 0:
            prec = num_correct / num_pred
        if num_gold > 0:
            rec = num_correct / num_gold
        if rec + prec > 0.0:
            f1 = 2 * rec * prec / (rec + prec)
        return {
            #"correct": num_correct,
            #"predicted": num_pred,
            #"gold": num_gold,
            "rec": rec,
            "prec": prec,
            "f1": f1
        }

    def evaluate(self, golds, preds):
        """
        https://github.com/MiuLab/SlotGated-SLU/blob/master/utils.py
        :param golds:
        :param preds:
        :return:
        """
        correct_chunk = {}
        correct_chunk_cnt = 0
        found_correct = {}
        found_correct_cnt = 0
        found_pred = {}
        found_pred_cnt = 0
        correct_tags = 0
        token_count = 0
        for correct_slot, pred_slot in zip(golds, preds):
            in_correct = False
            last_correct_tag = 'O'
            last_correct_type = ''
            last_pred_tag = 'O'
            last_pred_type = ''
            for c, p in zip(correct_slot, pred_slot):
                correct_tag, correct_type = self.split_tag(c)
                pred_tag, pred_type = self.split_tag(p)

                if in_correct:
                    if self.is_chunk_end(last_correct_tag, correct_tag,
                                         last_correct_type, correct_type) and \
                            self.is_chunk_end(last_pred_tag, pred_tag,
                                              last_pred_type, pred_type) and \
                            last_correct_type == last_pred_type:
                        in_correct = False
                        correct_chunk_cnt += 1
                        if last_correct_type in correct_chunk:
                            correct_chunk[last_correct_type] += 1
                        else:
                            correct_chunk[last_correct_type] = 1
                    elif self.is_chunk_end(last_correct_tag, correct_tag,
                                           last_correct_type, correct_type) != \
                            self.is_chunk_end(last_pred_tag, pred_tag,
                                              last_pred_type, pred_type) or \
                            correct_type != pred_type:
                        in_correct = False

                if self.is_chunk_start(last_correct_tag, correct_tag,
                                       last_correct_type, correct_type) and \
                        self.is_chunk_start(last_pred_tag, pred_tag,
                                            last_pred_type, pred_type) and \
                        correct_type == pred_type:
                    in_correct = True

                if self.is_chunk_start(last_correct_tag, correct_tag,
                                       last_correct_type, correct_type):
                    found_correct_cnt += 1
                    if correct_type in found_correct:
                        found_correct[correct_type] += 1
                    else:
                        found_correct[correct_type] = 1

                if self.is_chunk_start(last_pred_tag, pred_tag,
                                       last_pred_type, pred_type):
                    found_pred_cnt += 1
                    if pred_type in found_pred:
                        found_pred[pred_type] += 1
                    else:
                        found_pred[pred_type] = 1

                if correct_tag == pred_tag and correct_type == pred_type:
                    correct_tags += 1

                token_count += 1
                last_correct_tag = correct_tag
                last_correct_type = correct_type
                last_pred_tag = pred_tag
                last_pred_type = pred_type

            if in_correct:
                correct_chunk_cnt += 1
                if last_correct_type in correct_chunk:
                    correct_chunk[last_correct_type] += 1
                else:
                    correct_chunk[last_correct_type] = 1

        classes = set(found_correct) | set(found_pred) | set(correct_chunk)
        ret = {
            "overall": self.form_stats(
                correct_chunk_cnt,
                found_pred_cnt,
                found_correct_cnt
            )
        }
        if self.detailed:
            ret.update({
                "slots": {c: self.form_stats(
                    correct_chunk.get(c, 0),
                    found_pred.get(c, 0),
                    found_correct.get(c, 0),
                ) for c in classes},
            })
        return ret


def evaluate(golds, preds):
    # return ConllEvaluator().evaluate(golds, preds)
    return evaluate_conlleval(golds, preds)