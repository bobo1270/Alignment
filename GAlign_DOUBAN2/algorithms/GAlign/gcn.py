"""
This code was copied from the GCN implementation in DGL examples.
"""
import torch
import torch.nn as nn
from algorithms.GAlign.embedding_model import GCN


class GCN_dgi(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        # hidden layers
        for i in range(n_layers):
            self.layers.append(GCN(activation, in_feats, n_hidden))
            
        # output layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features,adj):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h,adj)
        return h