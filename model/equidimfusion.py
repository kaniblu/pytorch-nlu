"""
Fusing unknown number of inputs with equivalent dimensions
(unlike `fusion` which fuses fixed number of inputs with variable dimensions)
"""

from . import utils as mutils
from . import common
from . import fusion
from . import pooling
from . import nonlinear


class AbstractEquidimFusion(common.Module):
    def __init__(self, in_dim, out_dim):
        super(AbstractEquidimFusion, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward_loss(self, xs, lens=None):
        """
        :param xs: [batch_size, k, in_dim] Tensor
        :param lens: [batch_size] LongTensor
        :return:
        """
        raise NotImplementedError()


class PoolingEquidimFusion(AbstractEquidimFusion):
    name = "pooling-equidimfusion"

    def __init__(self, *args, pooling=pooling.AbstractPooling, **kwargs):
        super(PoolingEquidimFusion, self).__init__(*args, **kwargs)
        self.pooling_cls = pooling
        self.pooling = self.pooling_cls(
            dim=self.in_dim
        )
        self.output_layer = nonlinear.get_default()(
            in_dim=self.in_dim,
            out_dim=self.out_dim
        )

    def forward_loss(self, xs, lens=None):
        x = self.invoke(self.pooling, xs, lens)
        return self.invoke(self.output_layer, x)


class FusionNonlinearEquidimFusion(AbstractEquidimFusion):
    name = "fusion-equidimfusion"

    def __init__(self, *args, num_inputs=2,
                 fusion=fusion.AbstractFusion, **kwargs):
        super(FusionNonlinearEquidimFusion, self).__init__(*args, **kwargs)
        self.num_inputs = num_inputs
        self.fusion_cls = fusion
        self.fusion = self.fusion_cls(
            in_dims=[self.in_dim] * num_inputs,
            out_dim=self.in_dim
        )
        self.output_layer = nonlinear.get_default()(
            in_dim=self.in_dim,
            out_dim=self.out_dim
        )

    def forward_loss(self, xs, lens=None):
        x = self.invoke(self.fusion, *mutils.chop(xs, 1))
        return self.invoke(self.output_layer, x)