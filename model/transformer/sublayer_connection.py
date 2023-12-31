# -*- coding: utf-8 -*-
# date: 2018-11-30 15:17
import torch.nn as nn

from .layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout, device='cpu'):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        """
        return x + self.dropout(sublayer(self.norm(x)))
