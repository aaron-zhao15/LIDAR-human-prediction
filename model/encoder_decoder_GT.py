"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

from model.decoder_GT import Decoder_GPT
from model.encoder_GT import Encoder_GPT

# -----------------------------------------------------------------------------

class Encoder_Decoder_GPT(nn.Module):
    """ GPT Language Model """

    def __init__(self, n_layer, n_head, n_embd, vocab_size, block_size, pdrop=0.1, device='cpu'):
        super().__init__()
        assert vocab_size is not None
        assert block_size is not None
        self.block_size = block_size

        self.encoder = Encoder_GPT(n_layer, n_head, n_embd, vocab_size, block_size, pdrop, device)
        self.decoder = Decoder_GPT(n_layer, n_head, n_embd, vocab_size, block_size, pdrop, device)

    def forward(self, idx, targets=None):
        outputs_enc, _ = self.encoder(idx)

        # Decoder
        outputs_dec, _ = self.decoder(outputs_enc)

        return outputs_dec, outputs_enc

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        idx_tot = idx.clone().detach()
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:, :]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1:0, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            idx_next = logits
            # append sampled index to the running sequence and continue
            print(idx_tot.shape)
            idx = torch.cat((idx, idx_next), dim=1)
            idx_tot = torch.cat((idx_tot, idx_next), dim=1)
        return idx, idx_tot
    
class Encoder_Decoder_Classifier(nn.Module):
    """ GPT Language Model """

    def __init__(self, n_layer, n_head, n_embd, vocab_size, block_size, num_classes, pdrop=0.1, device='cpu'):
        super().__init__()
        assert vocab_size is not None
        assert block_size is not None
        self.block_size = block_size

        self.encoder = Encoder_GPT(n_layer, n_head, n_embd, vocab_size, block_size, pdrop, device)
        self.decoder = Decoder_GPT(n_layer, n_head, n_embd, vocab_size, block_size, pdrop, device)
        self.lm_classifier = nn.Linear(block_size*vocab_size, num_classes, bias=False, device=device)

    def forward(self, idx):
        outputs_enc, _ = self.encoder(idx)

        # Decoder
        outputs_dec, _ = self.decoder(outputs_enc)
        logits = self.lm_classifier(torch.flatten(outputs_dec, start_dim=1))

        return logits, outputs_enc
