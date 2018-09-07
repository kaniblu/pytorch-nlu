import torch
import torch.nn as nn

import utils
from . import common
from . import fusion
from . import utils as mutils


class MaskedSoftmax(nn.Module):
    def __init__(self, dim=0):
        super(MaskedSoftmax, self).__init__()
        self.dim = dim
        self.softmax = nn.Softmax(dim)

    def forward(self, x, mask=None):
        """
        Performs masked softmax, as simply masking post-softmax can be
        inaccurate
        :param x: [batch_size, num_items]
        :param mask: [batch_size, num_items]
        :return:
        """
        if mask is not None:
            mask = mask.float()
        if mask is not None:
            x_masked = x * mask + (1 - 1 / mask)
        else:
            x_masked = x
        x_max = x_masked.max(self.dim)[0]
        x_exp = (x - x_max.unsqueeze(self.dim)).exp()
        if mask is not None:
            x_exp = x_exp * mask.float()
        return x_exp / x_exp.sum(self.dim).unsqueeze(self.dim)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = MaskedSoftmax(2)

    def forward(self, qrys, keys, vals, num_keys=None):
        """
        Computes scaled dot product attention
        :param qrys: [batch_size, max_num_qrys, key_dim]
        :param keys: [batch_size, max_num_keys, key_dim]
        :param vals: [batch_size, max_num_keys, val_dim]
        :param num_keys: [batch_size] LongTensor
        :return: [batch_size, num_qrys, val_dim]
        """
        max_num_qrys, max_num_keys = qrys.size(1), keys.size(1)
        scale = 1 / qrys.new_full(tuple(), keys.size(2)).sqrt()
        # [batch_size, max_num_qrys, max_num_keys]
        logits = torch.bmm(qrys, keys.permute(0, 2, 1)) * scale
        if num_keys is not None:
            mask = mutils.mask(num_keys, max_len=max_num_keys)
            mask = mask.unsqueeze(1).expand_as(logits).contiguous()
        else:
            mask = None
        atts = self.softmax(logits, mask)
        return torch.bmm(atts, vals)


class AbstractAttention(common.Module):
    def __init__(self, qry_dim, val_dim):
        super(AbstractAttention, self).__init__()
        self.qry_dim, self.val_dim = qry_dim, val_dim

    def forward_loss(self, qry, vals, num_vals=None):
        raise NotImplementedError()


class MultiplicativeAttention(AbstractAttention):
    name = "multiplicative-attention"

    def __init__(self, *args, hidden_dim=100, **kwargs):
        super(MultiplicativeAttention, self).__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.q_linear = common.Linear(self.qry_dim, hidden_dim, bias=False)
        self.k_linear = common.Linear(self.val_dim, hidden_dim, bias=False)
        self.attend = ScaledDotProductAttention()

    def forward_loss(self, qry, vals, num_vals=None):
        qry = self.invoke(self.q_linear, qry)
        keys = self.invoke(self.k_linear, vals)
        return self.invoke(self.attend, qry, keys, vals, num_vals)


class AdditiveAttention(AbstractAttention):
    def __init__(self, *args, hidden_dim=100, **kwargs):
        super(AdditiveAttention, self).__init__(*args, **kwargs)
        self.linear = common.Linear(hidden_dim, 1, bias=False)
        self.hidden_dim = hidden_dim
        self.softmax = MaskedSoftmax(2)

    def combine(self, qry_exp, vals_exp):
        """
        :param qry_exp: [batch_size * num_qrys * num_vals, qry_dim]
        :param vals_exp: [batch_size * num_qrys * num_vals, val_dim]
        :return: [batch_size * num_qrys * num_vals, hidden_dim]
        """
        raise NotImplementedError()

    def forward_loss(self, qrys, vals, num_vals=None):
        batch_size = qrys.size(0)
        max_num_qrys, max_num_vals = qrys.size(1), vals.size(1)
        qrys_exp = qrys.unsqueeze(2)\
            .expand(batch_size, max_num_qrys, max_num_vals, self.qry_dim)
        vals_exp = vals.unsqueeze(1)\
            .expand(batch_size, max_num_qrys, max_num_vals, self.val_dim)
        qrys_exp = qrys_exp.contiguous().view(-1, self.qry_dim)
        vals_exp = vals_exp.contiguous().view(-1, self.val_dim)
        logits = self.invoke(self.linear, self.combine(qrys_exp, vals_exp))
        logits = logits.squeeze(-1)
        logits = logits.view(batch_size, max_num_qrys, max_num_vals)

        # handling for in-batch dynamic K
        if num_vals is not None:
            mask = mutils.mask(num_vals, max_num_vals)
            mask = mask.unsqueeze(1).expand_as(logits).contiguous()
        else:
            mask = None
        atts = self.softmax(logits, mask)
        yield "pass", torch.bmm(atts, vals)


class LinearAdditiveAttention(AdditiveAttention):
    name = "linear-additive-attention"

    def __init__(self, *args, **kwargs):
        super(LinearAdditiveAttention, self).__init__(*args, **kwargs)
        self.linear_qv = common.Linear(
            in_features=self.qry_dim + self.val_dim,
            out_features=self.hidden_dim
        )

    def combine(self, qry_exp, vals_exp):
        return self.invoke(self.linear_qv, torch.cat([qry_exp, vals_exp], 1))


class FusionAdditiveAttention(AdditiveAttention):
    name = "fuse-additive-attention"

    def __init__(self, *args, fusion=fusion.GatedSoftmaxFusion, **kwargs):
        super(FusionAdditiveAttention, self).__init__(*args, **kwargs)
        self.fuse_cls = fusion
        self.fuse = self.fuse_cls((self.qry_dim, self.val_dim), self.hidden_dim)

    def combine(self, qry_exp, vals_exp):
        return self.invoke(self.fuse, qry_exp, vals_exp)
