import torch
import torch.nn as nn


def roll_left(x, n=1):
    return torch.cat([x[:, n:], x[:, :n]], 1)


def roll_right(x, n=1):
    return torch.cat([x[:, -n:], x[:, :-n]], 1)


def shift_right(x, dim=1, n=1, fill="arithmetic"):
    size = x.size()
    pre_size, pos_size = size[:dim], size[dim + 1:]
    if fill == "arithmetic":
        pad = x[tuple(slice(None) for _ in pre_size) + (0, )]
        pad = pad.unsqueeze(dim).expand(*pre_size, n, *pos_size)
    elif fill == "roll":
        pad = x[:, -n:]
    elif fill == "zero":
        pad = x.new(*pre_size, n, *pos_size).zero_()
    else:
        raise ValueError(f"unrecognized fill type: {fill}")
    sx = x[tuple(slice(None) for _ in pre_size) + (slice(0, -n),)]
    return torch.cat([pad, sx], dim)


def mask(lens, max_len=None):
    if max_len is None:
        max_len = lens.max().item()
    enum = torch.range(0, max_len - 1).long()
    enum = enum.to(lens.device)
    enum_exp = enum.unsqueeze(0)
    return lens.unsqueeze(1) > enum_exp


def chop(xs, dim=0):
    return tuple(x.squeeze(dim) for x in torch.split(xs, 1, dim))


def embed_dot(emb, x):
    x_size = x.size()
    weight = emb.weight.t()
    o = torch.mm(x.view(-1, x_size[-1]), weight)
    return o.view(*x_size[:-1], -1)


class BeamSearchSequenceDecoder(object):

    """
    To make my life a little easier, this beam-search decoder assumes
    several things:

    1) The model we are dealing with is an rnn-like model with the following
       signature:

       input float-tensor [batch_size, max_len, input_dim], \
          (optional) lengths long-tensor [batch_size], \
          (optional) initial hidden float-tensor [batch_size, hidden_dim] ->
       hidden states float-tensor [batch_size, max_len, hidden_dim], \
          cell states float-tensor [batch_size, max_len, hidden_dim], \
          final-step hidden state float-tensor [batch_size, hidden_dim]

    2) Encoding and decoding functions that map from discrete indices to
       continuous representations and vice versa are provided with the following
       signatures:

       - encoder:

          discrete indices long-tensor [batch_size, max_len] ->
             latent rep. float-tensor [batch_size, max_len, input_dim]

       - decoder:

          latent rep. float-tensor [batch_size, max_len, hidden_dim] ->
             prob. distribution long-tensor [batch_size, max_len, num_labels]

    3) Assumes that bos and (optionally) eos labels are recognized by the
       label vocabulary and the rnn model. These are necessary for initial
       priming.

    Inherit this class and implement relevant methods and call `decode` method.
    """
    def __init__(self, bos, eos=None, maxlen=100, beam_size=3):
        self.bos = bos
        self.eos = eos
        self.maxlen = maxlen
        self.beam_size = beam_size
        self.max_float = 1e10
        self.min_float = -1e10
        self.softmax = nn.Softmax(2)

    def _rnn(self, w, lens=None, h=None):
        raise NotImplementedError()

    def _encode_discrete(self, w):
        raise NotImplementedError()

    def _decode_discrete(self, h):
        raise NotImplementedError()

    def decode(self, z):
        batch_size = z.size(0)
        # forces the beam searcher to search from the first index only
        # in the beginning
        x = z.new(batch_size, 1, 1).long().fill_(self.bos)
        has_eos = x.new(batch_size, 1).zero_().byte()
        probs = z.new(batch_size, 1).fill_(1.0)
        lens = x.new(batch_size, 1).fill_(1).long()
        while has_eos.prod().item() != 1 and lens.max() < self.maxlen:
            cur_beamsize, seq_len = x.size(1), x.size(2)
            x_emb = self._encode_discrete(x)
            x_emb = x_emb.view(batch_size * cur_beamsize, seq_len, -1)
            z_exp = z.unsqueeze(1).expand(batch_size, cur_beamsize, -1) \
                .contiguous().view(batch_size * cur_beamsize, -1)
            xo, _, _ = self._rnn(x_emb, lens.view(-1), z_exp)
            xo = xo[:, -1].view(batch_size, cur_beamsize, -1)
            logits = self._decode_discrete(xo)
            # for beams that already generated <eos>, prevent probability
            # depreciation.
            if self.eos is not None:
                eos_mask = has_eos.unsqueeze(-1).float()
                logits_eos = torch.full_like(logits, self.min_float)
                logits_eos[:, :, self.eos] = self.max_float
                logits = logits * (1 - eos_mask) + logits_eos * eos_mask
            # [batch_size x beam_size x vocab_size]
            p_vocab = probs.unsqueeze(-1) * self.softmax(logits)
            vocab_size = p_vocab.size(-1)
            # utilize 2d-flattened-to-1d indices
            probs, idx = torch.sort(p_vocab.view(batch_size, -1), 1, True)
            probs, idx = \
                probs[:, :self.beam_size], idx[:, :self.beam_size].long()
            beam_idx, preds = idx / vocab_size, idx % vocab_size
            x = torch.gather(x, 1,
                             beam_idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))
            x = torch.cat([x, preds.unsqueeze(-1)], 2)
            if self.eos is not None:
                has_eos = torch.gather(has_eos, 1, beam_idx)
                has_eos = (preds == self.eos) | has_eos
            lens = torch.gather(lens, 1, beam_idx)
            lens += (1 - has_eos).long()
        return x, lens + 1, probs


class BeamSearchDecoder(object):

    def __init__(self, logprobs, lens=None, beam_size=5):
        # [batch_size x max_len x num_labels] LongTensor
        self.logprobs = logprobs
        self.lens = lens
        self.beam_size = beam_size

        self.mask = None
        if self.lens is not None:
            self.mask = mask(lens, self.max_len)

    @property
    def batch_size(self):
        return self.logprobs.size(0)

    @property
    def max_len(self):
        return self.logprobs.size(1)

    @property
    def num_labels(self):
        return self.logprobs.size(2)

    def decode(self):
        # [batch_size x 1 x 1] LongTensor
        cum_logprobs, x = self.logprobs[:, 0].max(1)
        x = x.unsqueeze(1).unsqueeze(1)
        cum_logprobs = cum_logprobs.unsqueeze(1)

        for t in range(1, self.max_len):
            logprobs = self.logprobs[:, t]
            new_logprobs = cum_logprobs.unsqueeze(-1) + logprobs.unsqueeze(1)

            # utilize 2d-flattened-to-1d indices
            new_logprobs_flat, idx_flat = \
                torch.sort(new_logprobs.view(self.batch_size, -1), 1, True)
            new_logprobs_flat, idx_flat = \
                new_logprobs_flat[:, :self.beam_size], \
                idx_flat[:, :self.beam_size]
            beam_idx, preds = \
                idx_flat / self.num_labels, idx_flat % self.num_labels
            beam_idx_exp = beam_idx.unsqueeze(-1).expand(-1, -1, x.size(-1))
            x_indexed = torch.gather(x, 1, beam_idx_exp)
            x = torch.cat([x_indexed, preds.unsqueeze(-1)], 2)

            # write-back original logprobs for out-of-length items
            if self.mask is not None:
                mask = self.mask[:, t].float()
                old_logprobs = torch.gather(cum_logprobs, 1, beam_idx)
                new_logprobs = new_logprobs_flat * mask.unsqueeze(-1) + \
                               old_logprobs * (1 - mask.unsqueeze(-1))
            else:
                new_logprobs = new_logprobs_flat

            cum_logprobs = new_logprobs

        return x, cum_logprobs