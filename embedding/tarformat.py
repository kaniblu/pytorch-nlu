import io
import os
import pickle
import tarfile

from .common import Embeddings

import numpy as np


class TarFormatEmbeddings(Embeddings):

    name = "tar-format"

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.vocab = None
        self.array = None
        self._dim = None

    def preload(self):
        assert os.path.exists(self.path)
        with tarfile.TarFile(self.path, mode="r") as tf:
            self.vocab = pickle.load(tf.extractfile("vocab.pkl"))
            self.array = np.load(io.BytesIO(tf.extractfile("array.npy").read()))

    @property
    def dim(self):
        return self.array.shape[1]

    def __hash__(self):
        return hash(self.name) * 541 + hash(self.path)

    def __eq__(self, other):
        if not isinstance(other, TarFormatEmbeddings):
            return False
        return self.name == other.name and self.path == other.path

    def __contains__(self, item):
        return item in self.vocab

    def __getitem__(self, item):
        return self.array[self.vocab[item]]

    def __iter__(self):
        return iter((w, self.array[i]) for w, i in self.vocab.items())