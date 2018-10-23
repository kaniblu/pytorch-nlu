import subprocess

import numpy as np

from .common import Embeddings


class FastText(object):

    def __init__(self, fasttext_path, model_path, dtype=np.float32):
        self.fasttext_path = fasttext_path
        self.model_path = model_path
        self.args = [fasttext_path, "print-word-vectors", model_path]
        self.dtype = dtype
        self.process = None

    def query(self, word):
        self.process.stdin.write(f"{word}\n".encode())
        self.process.stdin.flush()
        line = self.process.stdout.readline().decode()
        line = " ".join(line.split()[1:])
        return np.fromstring(line, dtype=self.dtype, sep=" ")

    def __enter__(self):
        self.process = subprocess.Popen(
            args=self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.process.kill()


class FastTextEmbeddings(Embeddings):

    name = "fasttext"

    def __init__(self, fasttext_path, model_path):
        super(FastTextEmbeddings, self).__init__()
        self.fasttext_path = fasttext_path
        self.model_path = model_path
        self.fasttext = FastText(fasttext_path, model_path)

    def preload(self):
        self.fasttext.__enter__()

    def __hash__(self):
        return hash(self.name) * 6911 + \
               hash(self.fasttext_path) * 7043 + \
               hash(self.model_path) * 7919

    def __eq__(self, other):
        if not isinstance(other, FastTextEmbeddings):
            return False
        return self.fasttext_path == other.fasttext_path and \
               self.model_path == other.model_path

    def __getitem__(self, item):
        return self.fasttext.query(item)

    def __contains__(self, item):
        return True
