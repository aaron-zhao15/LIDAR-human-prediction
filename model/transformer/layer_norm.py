# -*- coding: utf-8 -*-
# date: 2018-11-29 20:14
import torch

import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Construct a layernorm module (See citation for details).
    """

    def __init__(self, features, eps=1e-6, device='cpu'):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features, device=device))
        self.b_2 = nn.Parameter(torch.zeros(features, device=device))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
