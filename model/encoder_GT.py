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

# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class BertBlock(nn.Module):
    """ Feed forward block """

    def __init__(self, n_head, n_embd, pdrop=0.1, device='cpu'):
        super().__init__()
        self.attention = nn.MultiheadAttention(n_embd, n_head, device=device)
        self.feed_forward = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd, device=device),
            c_proj  = nn.Linear(4 * n_embd, n_embd, device=device),
            act     = NewGELU(),
            dropout = nn.Dropout(pdrop),
        ))
        self.input_sublayer = nn.ModuleDict(dict(
            norm    = nn.LayerNorm(n_embd, device=device),
            dropout = nn.Dropout(pdrop)
        ))
        self.output_sublayer = nn.ModuleDict(dict(
            norm    = nn.LayerNorm(n_embd, device=device),
            dropout = nn.Dropout(pdrop)
        ))
        self.dropout = nn.Dropout(p=pdrop)
        m = self.feed_forward
        self.feed_forward_f = lambda _x: m.dropout(m.c_proj(m.act(m.c_fc(_x))))
        self.layer_f = lambda _x, forward, sublayer: _x + sublayer.dropout(forward(sublayer.norm(_x)))

    def forward(self, x):
        x = self.layer_f(x, lambda _x: self.attention.forward(_x, _x, _x)[0], self.input_sublayer)
        x = self.layer_f(x, self.feed_forward_f, self.output_sublayer)
        return self.dropout(x)

class Encoder_GPT(nn.Module):
    """ GPT Language Model """

    def __init__(self, n_layer, n_head, n_embd, vocab_size, block_size, pdrop=0.1, device='cpu'):
        super().__init__()
        assert vocab_size is not None
        assert block_size is not None
        self.block_size = block_size

        self.transformer = nn.ModuleDict(dict(
            # wte = nn.Embedding(vocab_size, n_embd, device=device),
            # not really an embedding but a linear layer to help match shapes
            wte = nn.Linear(vocab_size, n_embd, device=device),
            wpe = nn.Embedding(block_size, n_embd, device=device),
            # this uses the positional embedding specified in https://arxiv.org/pdf/2003.08111.pdf
            # wpe = PositionalEncoding(d_model=n_embd, dropout=0.1, max_len=block_size, device=device),
            drop = nn.Dropout(pdrop),
            h = nn.ModuleList([BertBlock(n_head, n_embd, pdrop=0.1, device=device) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd, device=device),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False, device=device)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t, s = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

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


class Encoder_GPT_classifier(nn.Module):
    """ GPT Language Model """

    def __init__(self, n_layer, n_head, n_embd, vocab_size, block_size, num_classes, pdrop=0.1, device='cpu'):
        super().__init__()
        assert vocab_size is not None
        assert block_size is not None
        self.block_size = block_size
        self.transformer = nn.ModuleDict(dict(
            # wte = nn.Embedding(vocab_size, n_embd, device=device),
            # not really an embedding but a linear layer to help match shapes
            wte = nn.Linear(vocab_size, n_embd, device=device),
            wpe = nn.Embedding(block_size, n_embd, device=device),
            # this uses the positional embedding specified in https://arxiv.org/pdf/2003.08111.pdf
            # wpe = PositionalEncoding(d_model=n_embd, dropout=0.1, max_len=block_size, device=device),
            drop = nn.Dropout(pdrop),
            h = nn.ModuleList([BertBlock(n_head, n_embd, pdrop=0.1, device=device) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd, device=device),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False, device=device)
        # self.lm_classifier = nn.Linear(n_embd, num_classes, bias=False, device=device)
        self.lm_classifier = nn.Sequential(nn.Linear(n_embd, n_embd, device=device),
                                nn.Dropout(0.5),
                                nn.Tanh(),
                                nn.Linear(n_embd, num_classes, device=device))
        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx):
        device = idx.device
        b, t, s = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        encoder_fw = self.lm_head(x)
        cls_output = x[:, -1, :]
        logits = self.lm_classifier(cls_output)
        return logits, encoder_fw
    
"""
Action Recognition Models
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from model.configs import load_model, load_embedding_fn


class BiLSTM(nn.Module):
    def __init__(self, model_params, device=None):
        super().__init__()

        input_dim = model_params['input_dim']
        hidden_dim = model_params['lstm_hidden']
        num_layers = model_params['num_layers']
        lstm_dropout = model_params['lstm_dropout']
        fc_dim = model_params['fc_dim']
        num_cls = model_params['num_classes']

        # BiLSTM
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                              dropout=lstm_dropout, bidirectional=True, device=device)

        # Logit layer
        self.fc = nn.Sequential(nn.Linear(2 * hidden_dim, fc_dim, device=device),
                                nn.Dropout(0.5),
                                nn.Tanh(),
                                nn.Linear(fc_dim, num_cls, device=device))
        # Params
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, x_input, x_seq_len):
        # x_input: [batch_size, seq_len, input_dim]

        # x = pack_padded_sequence(x_input, x_seq_len, batch_first=True, enforce_sorted=False)
        x = x_input
        outputs, (hidden, cell) = self.bilstm(x)        # outputs: [sum_{i=0}^batch (seq_lens[i]), 2 * hidden_dim]

        # hidden: [num_layers * 2, batch_size, hidden_dim]
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)   # [num_layers, 2, batch_size, hidden_dim]

        # Skip hidden states of intermediate layers
        hidden = hidden[-1]                                             # [2, batch_size, hidden_dim]

        # Extract the forward & backward hidden states
        forward_h = hidden[0]
        backward_h = hidden[1]

        # Concatenate hidden states
        final_hidden = torch.cat([forward_h, backward_h], dim=1)       # [batch_size, 2*hidden_dim]

        logits = self.fc(final_hidden)                                 # [batch_size, num_cls]

        return logits, None


# ---------------------------------------------------------------
class Transformer(nn.Module):
    """
    Adapts HuggingFace's Transformer for handling video embeddings
    """
    def __init__(self, model_params, device=None):
        super(Transformer, self).__init__()

        input_dim = model_params['input_dim']
        model_name = model_params['model_name']         # e.g. bert, roberta, etc.
        config_name = model_params['config_name']       # e.g. bert-base-uncased, roberta-base, etc.
        config_dict = model_params['config_dict']       # custom config params
        use_pretrained = model_params['use_pretrained']
        fc_dim = model_params['fc_dim']
        num_cls = model_params['num_classes']

        self.max_len = model_params['max_video_len']

        self.device = device

        # Load transformer for the given name & config
        self.transformer = load_model(model_name, config_dict, config_name, use_pretrained)

        hidden_dim = self.get_hidden_size()

        # Project video embedding to transformer dim
        self.projection_layer = nn.Linear(input_dim, hidden_dim, device=device)

        # Load the embedding function for encoding token ids
        self.embedding_fn = load_embedding_fn(model_name, config_name)

        # Logit layer
        self.fc = nn.Sequential(nn.Linear(hidden_dim, fc_dim, device=device),
                                nn.Dropout(0.5),
                                nn.Tanh(),
                                nn.Linear(fc_dim, num_cls, device=device))

    def forward(self, video_emb, token_seq_ids, attention_mask):
        """
        # max_seq_len = max_video_len + num_special_tokens

        :param video_emb: [batch, max_video_len, video_emb_dim]
        :param token_seq_ids: [batch, max_seq_len]
        :param attention_mask: [batch, max_seq_len] <br>
        """
        # Project video embedding to token embedding space (hidden dim)
        video_emb = self.projection_layer(video_emb)

        # Encode video with positional embeddings
        video_emb = self.embedding_fn(inputs_embeds=video_emb,
                                      position_ids=torch.arange(1, self.max_len + 1, device=self.device))

        # Encode token sequence ([CLS] [UNK].. [SEP] [PAD]..)
        embeddings_input = self.embedding_fn(input_ids=token_seq_ids)

        # Replace [UNK] embeddings with video embeddings
        embeddings_input[:, 1: self.max_len+1, :] = video_emb

        # Extract the sequence embeddings from the final layer of the transformer
        last_hidden_states = self.transformer(inputs_embeds=embeddings_input,           # [batch, max_len, emb_dim]
                                              attention_mask=attention_mask)[0]

        # Obtain the CLS token embedding from the last hidden layer
        cls_output = last_hidden_states[:, 0, :]                                        # [batch, emb_dim]

        logits = self.fc(cls_output)

        return logits

    def get_hidden_size(self):
        return self.transformer.config.hidden_size
