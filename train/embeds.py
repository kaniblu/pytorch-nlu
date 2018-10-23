import torch

import utils
import embedding.fasttext
import embedding.tarformat
import embedding.glove


def add_embed_arguments(parser):
    parser.add_argument("--word-embed-type", type=str, default=None)
    parser.add_argument("--word-embed-path", type=str, default=None)
    parser.add_argument("--fasttext-path", type=str,
                        help="path to FastText executable binary")


def _load_embeddings(module, vocab, we):
    for w in utils.tqdm(vocab.f2i, desc="loading word embeddings"):
        if w not in we:
            continue
        v = we[w]
        idx = vocab.f2i[w]
        module.load(idx, torch.FloatTensor(v))


def get_embeddings(args, vocab=None):
    return utils.map_val(args.word_embed_type, {
        "glove-format": lambda: embedding.glove.GloveFormatEmbeddings(
            path=args.word_embed_path,
            words=set(vocab.f2i) if vocab is not None else None
        ),
        "tar-format": lambda: embedding.tarformat.TarFormatEmbeddings(
            path=args.word_embed_path
        ),
        "fasttext": lambda: embedding.fasttext.FastTextEmbeddings(
            fasttext_path=args.fasttext_path,
            model_path=args.word_embed_path
        )
    }, "embedding type")()


def load_embeddings(args, vocab, module):
    if args.word_embed_type is None:
        return
    embeddings = get_embeddings(args, vocab)
    embeddings.preload()
    _load_embeddings(module, vocab, embeddings)
