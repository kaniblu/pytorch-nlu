from . import common
from . import rnncell
from . import nonlinear


class AbstractRNN(common.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(AbstractRNN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

    def forward_loss(self, x, lens=None, h0=None):
        raise NotImplementedError()


class RNN(AbstractRNN):
    name = "rnn"

    def __init__(self, *args, cell=rnncell.AbstractRNNCell, **kwargs):
        super(RNN, self).__init__(*args, **kwargs)
        self.cell_cls = cell
        self.input_layer = nonlinear.get_default()(
            in_dim=self.in_dim,
            out_dim=self.in_dim
        )
        self.cell = self.cell_cls(
            input_dim=self.in_dim,
            hidden_dim=self.hidden_dim
        )
        self.output_layer = nonlinear.get_default()(
            in_dim=self.hidden_dim,
            out_dim=self.out_dim
        )

    def forward_loss(self, x, lens=None, h0=None):
        x = self.invoke(self.input_layer, x)
        o, c, h = self.invoke(self.cell, x, lens, h0)
        o = self.invoke(self.output_layer, o)
        return o, c, h