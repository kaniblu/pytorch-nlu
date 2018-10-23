import torch.nn as nn

import torchmodels
from torchmodels.modules import activation


class AbstractClassifier(torchmodels.Module):

    def __init__(self, hidden_dim, num_labels):
        super(AbstractClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels

    def forward(self, h):
        raise NotImplementedError()


class MLPClassifier(AbstractClassifier):

    name = "mlp-classifier"

    def __init__(self, *args, num_hidden_layers=1,
                 activation=activation.AbstractActivation,
                 dropout=0.0, batch_norm=False, **kwargs):
        super(MLPClassifier, self).__init__(*args, **kwargs)
        self.num_hidden_layers = num_hidden_layers
        self.activation_cls = activation
        self.dropout_prob = dropout
        self.should_dropout = dropout > 0.0
        self.should_batchnorm = batch_norm

        layers = []
        for _ in range(self.num_hidden_layers):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            if self.should_batchnorm:
                layers.append(nn.BatchNorm1d(self.hidden_dim))
            layers.append(self.activation_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(self.hidden_dim, self.num_labels))
        self.sequential = nn.Sequential(*layers)

    def forward(self, h):
        return self.sequential(h)