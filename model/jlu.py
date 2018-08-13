import torch
import torch.nn as nn
import torch.nn.functional as nf
import torch.nn.init as ninit

from . import common
from . import embedding
from . import equidimfusion
from . import rnn
from . import rnncell
from . import attention
from . import pooling
from . import nonlinear
from . import utils as mutils


class AbstractJointLU(common.Module):
    def __init__(self, hidden_dim, word_dim, label_dim, intent_dim,
                 num_words, num_labels, num_intents):
        super(AbstractJointLU, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_dim = word_dim
        self.label_dim = label_dim
        self.intent_dim = intent_dim
        self.num_words = num_words
        self.num_labels = num_labels
        self.num_intents = num_intents
        self.word_dims = [word_dim, label_dim, intent_dim]
        self.vocab_sizes = [num_words, num_labels, num_intents]

    def embeddings(self):
        raise NotImplementedError()

    def forward_loss(self, w, l, lens):
        raise NotImplementedError()

    def predict(self, w, lens, label_bos):
        raise NotImplementedError()


class SimpleLU(AbstractJointLU):
    name = "simple-jlu"

    def __init__(self, *args, word_embed=embedding.AbstractEmbedding, **kwargs):
        super(SimpleLU, self).__init__(*args, **kwargs)
        self.word_embed_cls = word_embed
        self.word_embed = self.word_embed_cls(
            vocab_size=self.num_words,
            dim=self.word_dim
        )
        self.lstm = nn.LSTM(
            input_size=self.word_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.attend = attention.ScaledDotProductAttention()
        self.label_output = nn.Linear(self.hidden_dim * 2, self.num_labels + 1)
        self.intent_output = nn.Linear(self.hidden_dim * 2, self.num_intents + 1)

    def embeddings(self):
        return [self.word_embed, None, None]

    def forward_loss(self, w, l, lens):
        w_emb = self.invoke(self.word_embed, w)
        whs, (wh, _) = self.lstm(w_emb)
        wh = wh.permute(1, 0, 2).contiguous()
        wh = wh.view(-1, self.hidden_dim * 2)
        lgts = self.label_output(whs)
        igts = self.intent_output(wh)
        return lgts, igts

    def predict(self, w, lens, label_bos):
        lgts, igts = self.forward_loss(w, None, lens)
        lprobs = nf.softmax(lgts, 2)
        iprobs = nf.softmax(igts, 1)
        return (lgts.max(2)[1], igts.max(1)[1]), (lprobs, iprobs)


class RNNJointLU(AbstractJointLU):
    name = "rnn-jlu"

    def __init__(self, *args, dropout_prob=0.5, dropout="outer",
                 word_embed=embedding.AbstractEmbedding,
                 label_embed=embedding.AbstractEmbedding,
                 intent_embed=embedding.AbstractEmbedding,
                 pooling=pooling.AbstractPooling,
                 rnn_cell=rnncell.AbstractRNNCell, **kwargs):
        super(RNNJointLU, self).__init__(*args, **kwargs)
        self.dropout_prob = dropout_prob
        self.word_embed_cls = word_embed
        self.label_embed_cls = label_embed
        self.intent_embed_cls = intent_embed
        self.pooling_cls = pooling
        self.rnn_cls = rnn_cell
        self.dropout_loc = dropout
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.word_embed = self.word_embed_cls(
            vocab_size=self.num_words,
            dim=self.word_dim
        )
        self.label_embed = self.label_embed_cls(
            vocab_size=self.num_labels,
            dim=self.label_dim
        )
        self.intent_embed = self.intent_embed_cls(
            vocab_size=self.num_intents,
            dim=self.intent_dim
        )
        self.rnn = self.rnn_cls(
            input_dim=self.word_dim,
            hidden_dim=self.hidden_dim
        )
        self.label_layer = nonlinear.get_default()(
            in_dim=self.hidden_dim,
            out_dim=self.label_dim
        )
        self.intent_pooling = self.pooling_cls(
            dim=self.hidden_dim
        )
        self.intent_layer = nonlinear.get_default()(
            in_dim=self.hidden_dim,
            out_dim=self.intent_dim
        )

    def embeddings(self):
        return [self.word_embed, self.label_embed, self.intent_embed]

    def forward_loss(self, w, l, lens):
        w_emb = self.invoke(self.word_embed, w)
        whs, _, _ = self.invoke(self.rnn, w_emb, lens)
        if self.dropout_loc in {"inner", "both"}:
            whs = self.dropout(whs)
        lhs = self.invoke(self.label_layer, whs)
        wh = self.invoke(self.intent_pooling, whs)
        ih = self.invoke(self.intent_layer, wh)
        if self.dropout_loc in {"outer", "both"}:
            lhs = self.dropout(lhs)
            ih = self.dropout(ih)
        lgts = mutils.embed_dot(self.label_embed, lhs)
        igts = mutils.embed_dot(self.intent_embed, ih)
        return lgts, igts

    def predict(self, w, lens, label_bos):
        lgts, igts = self.forward_loss(w, None, lens)
        lprobs = nf.softmax(lgts, 2)
        iprobs = nf.softmax(igts, 1)
        return (lgts.max(2)[1], igts.max(1)[1]), (lprobs, iprobs)


class AttentiveJointLU(AbstractJointLU):
    name = "attentive-jlu"

    def __init__(self, *args, beam_size=10,
                 word_embed=embedding.AbstractEmbedding,
                 label_embed=embedding.AbstractEmbedding,
                 intent_embed=embedding.AbstractEmbedding,
                 word_rnn=rnncell.AbstractRNNCell,
                 label_rnn=rnncell.AbstractRNNCell,
                 intent_network=equidimfusion.AbstractEquidimFusion,
                 attention=attention.AbstractAttention, **kwargs):
        super(AttentiveJointLU, self).__init__(*args, **kwargs)
        self.beam_size = beam_size
        self.word_embed_cls = word_embed
        self.label_embed_cls = label_embed
        self.intent_embed_cls = intent_embed
        self.word_rnn_cls = word_rnn
        self.label_rnn_cls = label_rnn
        self.intent_network_cls = intent_network
        self.attention_cls = attention
        self.word_embed = self.word_embed_cls(
            vocab_size=self.num_words,
            dim=self.word_dim
        )
        self.label_embed = self.label_embed_cls(
            vocab_size=self.num_labels,
            dim=self.label_dim
        )
        self.intent_embed = self.intent_embed_cls(
            vocab_size=self.num_intents,
            dim=self.intent_dim
        )
        self.word_rnn = self.word_rnn_cls(
            input_dim=self.word_dim,
            hidden_dim=self.hidden_dim,
        )
        self.label_rnns = common.ModuleList([
            self.label_rnn_cls(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
            ),
            self.label_rnn_cls(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
            )
        ])
        self.intent_fusion = self.intent_network_cls(
            in_dim=self.hidden_dim,
            out_dim=self.hidden_dim
        )
        self.label_attention = self.attention_cls(
            qry_dim=self.hidden_dim,
            val_dim=self.hidden_dim
        )
        self.intent_qry = common.Parameter(torch.zeros(self.hidden_dim),
                                           requires_grad=True)
        self.intent_attention = self.attention_cls(
            qry_dim=self.hidden_dim,
            val_dim=self.hidden_dim
        )
        self.intent_nonlinear = nonlinear.get_default()(
            in_dim=self.hidden_dim + self.hidden_dim,
            out_dim=self.intent_dim
        )

    def embeddings(self):
        return [self.word_embed, self.label_embed, self.intent_embed]

    def reset_parameters(self):
        super(AttentiveJointLU, self).reset_parameters()
        ninit.normal_(self.intent_qry.data.detach())

    def forward_label(self, whs, wh, l, lens):
        """
        Performs label inference only, given hidden outputs from word
        rnn and ground truth labels.
        :param whs: [batch_size, max_seq_len, word_dim] Tensor
        :param l: [batch_size, max_seq_len, label_dim] Tensor
        :param lens: [batch_size] LongTensor
        :return: [batch_size, max_seq_len, label_dim] Tensor
        """
        li1 = torch.cat([l, whs], 2)
        lhs, _, lh = self.invoke(self.label_rnns[0], li1, lens, wh)
        lcs = self.invoke(self.label_attention, lhs, whs, lens)
        li2 = torch.cat([whs, lcs, l], 2)
        los, _, _ = self.invoke(self.label_rnns[1], li2, lens, wh)
        return los

    def forward_intent(self, whs, lens):
        """
        Performs intent inference
        :param whs: [batch_size, max_seq_len, word_dim] Tensor
        :param lens: [batch_size] LongTensor
        :return: [batch_size, intent_dim]
        """
        ic_qry = self.intent_qry\
            .unsqueeze(0).unsqueeze(0).expand(whs.size(0), 1, self.hidden_dim)
        ic = self.invoke(self.intent_attention, ic_qry, whs, lens).squeeze(1)
        ih = self.invoke(self.intent_fusion, whs, lens)
        io = self.invoke(self.intent_nonlinear, torch.cat([ih, ic], 1))
        return io

    def forward_loss(self, w, l, lens):
        w = self.invoke(self.word_embed, w)
        l = self.invoke(self.label_embed, l)
        l = mutils.shift_right(l)
        whs, _, wh = self.invoke(self.word_rnn, w, lens)

        # label inference
        los = self.forward_label(whs, wh, l, lens)
        lgts = mutils.embed_dot(self.label_embed, los) # label logits

        # intent inference
        io = self.forward_intent(whs, lens)
        igts = mutils.embed_dot(self.intent_embed, io) # intent logits

        return lgts, igts

    def predict(self, w, lens, label_bos):
        batch_size = w.size(0)
        softmax = nn.Softmax(1)
        w = self.invoke(self.word_embed, w)
        whs, _, wh = self.invoke(self.word_rnn, w, lens)
        def _forward(x, lens=None, h0=None):
            seqlen = x.size(1)
            beam_size = x.size(0) // batch_size
            whs_exp = whs[:, :x.size(1)]\
                .unsqueeze(1).expand(batch_size, beam_size, -1, -1)
            whs_exp = whs_exp.contiguous().view(-1, seqlen, self.hidden_dim)
            o = self.forward_label(whs_exp, h0, x, lens)
            return o, None, None
        bsdec = mutils.BeamsearchDecoder(
            rnn=_forward,
            emb=self.label_embed,
            bos=label_bos,
            maxlen=w.size(1) + 1,
            beam_size=self.beam_size
        )
        labels, _, lprobs = bsdec.decode(wh)
        labels, lprobs = labels[:, 0, 1:], lprobs[:, 0]
        io = self.forward_intent(whs, lens)
        igts = mutils.embed_dot(self.intent_embed, io)
        iprobs = softmax(igts)
        intents = iprobs.max(1)[1]
        return (labels, intents), (lprobs, iprobs)


class SlotGatedJointLU(AbstractJointLU):
    name = "slotgated-jlu"

    def __init__(self, *args,
                 word_embed=embedding.AbstractEmbedding,
                 label_embed=embedding.AbstractEmbedding,
                 intent_embed=embedding.AbstractEmbedding,
                 word_rnn=rnncell.AbstractRNNCell,
                 label_attention=attention.AbstractAttention,
                 intent_attention=attention.AbstractAttention, **kwargs):
        super(SlotGatedJointLU, self).__init__(*args, **kwargs)
        self.word_embed_cls = word_embed
        self.label_embed_cls = label_embed
        self.intent_embed_cls = intent_embed
        self.word_rnn_cls = word_rnn
        self.label_attention_cls = label_attention
        self.intent_attention_cls = intent_attention
        self.word_embed = self.word_embed_cls(
            vocab_size=self.num_words,
            dim=self.word_dim
        )
        self.label_embed = self.label_embed_cls(
            vocab_size=self.num_labels,
            dim=self.label_dim
        )
        self.intent_embed = self.intent_embed_cls(
            vocab_size=self.num_intents,
            dim=self.intent_dim
        )
        self.word_rnn = self.word_rnn_cls(
            input_dim=self.word_dim,
            hidden_dim=self.hidden_dim,
        )
        self.label_proj = common.Linear(
            in_features=self.hidden_dim + self.hidden_dim,
            out_features=self.label_dim
        )
        self.intent_proj = common.Linear(
            in_features=self.hidden_dim + self.hidden_dim,
            out_features=self.intent_dim
        )
        self.gate_vector = common.Parameter(torch.zeros(self.hidden_dim),
                                            requires_grad=True)
        self.gate_proj = common.Linear(
            in_features=self.hidden_dim,
            out_features=self.hidden_dim,
            bias=False
        )
        self.label_attention = self.label_attention_cls(
            qry_dim=self.hidden_dim,
            val_dim=self.hidden_dim
        )
        self.intent_attention = self.intent_attention_cls(
            qry_dim=self.hidden_dim,
            val_dim=self.hidden_dim
        )
        self.tanh = nn.Tanh()

    def embeddings(self):
        return [self.word_embed, self.label_embed, self.intent_embed]

    def reset_parameters(self):
        super(SlotGatedJointLU, self).reset_parameters()
        ninit.normal_(self.gate_vector.data.detach())

    def get_slot_gate(self, lhs, ih):
        x = self.tanh(lhs + self.gate_proj(ih).unsqueeze(1))
        x = x * self.gate_vector
        return x.sum(2)

    def forward_loss(self, w, l, lens):
        w = self.invoke(self.word_embed, w)
        whs, _, wh = self.invoke(self.word_rnn, w, lens)
        ih = self.invoke(self.intent_attention, wh.unsqueeze(1), whs, lens)
        ih = ih.squeeze(1)
        lhs = self.invoke(self.label_attention, whs, whs, lens)
        g = self.get_slot_gate(lhs, ih)
        lhs = torch.cat([whs, lhs * g.unsqueeze(-1)], 2)
        lhs = self.invoke(self.label_proj, lhs)
        ih = torch.cat([wh, ih], 1)
        ih = self.invoke(self.intent_proj, ih)
        lgts = mutils.embed_dot(self.label_embed, lhs) # label logits
        igts = mutils.embed_dot(self.intent_embed, ih) # intent logits
        return lgts, igts

    def predict(self, w, lens, label_bos):
        lgts, igts = self.forward_loss(w, None, lens)
        lprobs = nf.softmax(lgts, 2)
        iprobs = nf.softmax(igts, 1)
        return (lgts.max(2)[1], igts.max(1)[1]), (lprobs, iprobs)