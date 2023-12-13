import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.decoder import Decoder
from transformer.multihead_attention import MultiHeadAttention
from transformer.positional_encoding import PositionalEncoding
from transformer.pointerwise_feedforward import PointerwiseFeedforward
from transformer.encoder_decoder import EncoderDecoder
from transformer.encoder import Encoder
from transformer.encoder_layer import EncoderLayer
from transformer.decoder_layer import DecoderLayer
from transformer.batch import subsequent_mask
import numpy as np
import scipy.io
import os

import copy
import math

class IndividualTF(nn.Module):
    def __init__(self, enc_inp_size, dec_inp_size, dec_out_size, N=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1,mean=[0,0],std=[0,0], device='cpu'):
        super(IndividualTF, self).__init__()
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model, device=device)
        ff = PointerwiseFeedforward(d_model, d_ff, dropout, device=device)
        position = PositionalEncoding(d_model, dropout, device=device)
        self.mean=np.array(mean)
        self.std=np.array(std)
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout, device=device), N, device=device),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout, device=device), N, device=device),
            nn.Sequential(LinearEmbedding(enc_inp_size,d_model, device=device), c(position)),
            nn.Sequential(LinearEmbedding(dec_inp_size,d_model, device=device), c(position)),
            Generator(d_model, dec_out_size, device=device))    
        self.device = device

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        dec_inp = torch.ones((x.shape[0], 1, (x.shape[2]//2)//3)).to(self.device).float()
        src_att = torch.ones((x.shape[0], 1, x.shape[1])).to(self.device).float()
        trg_att = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0],1,1).to(self.device).float()
        return self.model.generator(self.model(x, dec_inp, src_att, trg_att))

class LinearEmbedding(nn.Module):
    def __init__(self, inp_size, d_model, device='cpu'):
        super(LinearEmbedding, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(inp_size, d_model, device=device)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, out_size, device='cpu'):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, out_size, device=device)

    def forward(self, x):
        return self.proj(x)


