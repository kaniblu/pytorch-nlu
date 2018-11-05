import sys
import json
import argparse

import utils
from . import evaluate


parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument("--label-path", required=True)
parser.add_argument("--intent-path", required=True)
parser.add_argument("--gold-label-path", required=True)
parser.add_argument("--gold-intent-path", required=True)
parser.add_argument("--format", default="yaml",
                    choices=["yaml", "json"])
parser.add_argument("--detailed", action="store_true", default=False)


def load(path):
    with open(path, "r") as f:
        return [line.rstrip() for line in f]


def eval(args):
    lpreds, ipreds = load(args.label_path), load(args.intent_path)
    lgolds, igolds = \
        load(args.gold_label_path), load(args.gold_intent_path)
    res = evaluate(lgolds, igolds, lpreds, ipreds, detailed=args.detailed)
    dump = utils.map_val(args.format, {
        "yaml": utils.dump_yaml,
        "json": json.dump,
    }, "output format")
    dump(res, sys.stdout)


if __name__ == "__main__":
    eval(parser.parse_known_args()[0])
