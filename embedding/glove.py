import io
import os
import gzip
import logging

import numpy as np

import utils
from .common import Embeddings


class GloveFormatReader(utils.UniversalFileReader):

    def open_txt(self, path):
        return open(path, "r")

    def open_gz(self, path):
        return io.TextIOWrapper(gzip.open(path, "r"))


class GloveFormatEmbeddings(Embeddings):

    name = "glove-format"

    def __init__(self, path, dim=300, words=None):
        self.path = os.path.abspath(path)
        self.data = None
        self._dim = dim
        self.vocab = words

    @staticmethod
    def _tqdm(iterable=None):
        return utils.tqdm(
            iterable=iterable,
            desc="loading glove",
            unit="w",
        )

    def preload(self):
        self.data = {}
        dim = self.dim
        reader = GloveFormatReader(default_ext="txt")
        with reader(self.path) as f:
            for line in utils._tqdm.tqdm(f):
                tokens = line.split()
                word = " ".join(tokens[:-dim])
                if self.vocab is not None and word not in self.vocab:
                    continue
                vec = np.array([float(v) for v in tokens[-dim:]])
                self.data[word] = vec

        loaded_words = set(self.data.keys())
        stats = {"num-words": len(self.data)}
        if self.vocab is not None:
            stats["coverage"] = len(loaded_words & self.vocab) / len(self.vocab)
        stats = {k: f"{v:.4f}" for k, v in stats.items()}
        logging.info(f"{self.name} embeddings from {self.path} loaded,"
                     f" {utils.join_dict(stats, ', ', '=')}")

    def __hash__(self):
        return hash(self.name) * 541 + hash(self.path)

    def __eq__(self, other):
        if not isinstance(other, GloveFormatEmbeddings):
            return False
        return self.name == other.name and self.path == other.path

    @property
    def dim(self):
        return self._dim

    def __contains__(self, item):
        return item in self.data

    def __getitem__(self, item):
        return self.data[item]

    def __iter__(self):
        return iter(self.data.items())