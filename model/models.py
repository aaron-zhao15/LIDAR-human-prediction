import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import pdb
import math

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class GRU_model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, drop_prob=0.0):
        super(GRU_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # self.embedding = nn.Embedding(input_size, hidden_dim)
        self.gru = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        joint_dims = x.size(2)//2
        #Initializing hidden state for first input using method defined below
        h = self.init_hidden(batch_size)
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out))
        out = out.reshape((batch_size, seq_len, joint_dims))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden
    
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, seq_len):
        super(DecoderRNN, self).__init__()
        self.seq_len = seq_len
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(self.seq_len):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden
    
class RNN_model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN_model, self).__init__()

        # Defining some parameters
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True, dropout=0.2)
        # self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True, dropout=0.2)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x, target):
        
        batch_size = x.size(0)
        seq_len = x.size(1)

        #Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # print(hidden.shape)
        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        out = self.relu(out)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        out = out.reshape((batch_size, seq_len, self.output_size))
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden

class Encoder_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, rnn_unit, residual=False, out_dropout=0, std_mask=False,
                 veloc=False, pos_embed=False, pos_embed_dim=96, device="cuda"):
        super(Encoder_Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.residual = residual
        self.dropout_out = nn.Dropout(p=out_dropout).to(device)
        self.linear = nn.Linear(hidden_size, input_size).to(device)
        self.std_mask = std_mask
        self.veloc = veloc
        rnn_input = 2*input_size if veloc else input_size

        self.pos_embed = pos_embed
        if pos_embed:
            self.position_embeding = position_embedding(d_model=pos_embed_dim)
            rnn_input = rnn_input + self.position_embeding.size(1)
            self.position_embeding = self.position_embeding.to(device)

        if rnn_unit == 'gru':
            self.rnn = nn.GRU(rnn_input, hidden_size, num_layers=num_layer).to(device)
        if rnn_unit == 'lstm':
            self.rnn = nn.LSTM(rnn_input, hidden_size, num_layers=num_layer).to(device)

    def forward_seq(self, input, hidden=None):
        pred_pre = input[:,:,0:self.input_size].clone()
        if hidden is None:
            output, hidden_state = self.rnn(input)
        else:
            # print(input.dtype, hidden.dtype)
            output, hidden_state = self.rnn(input, hidden)
        pred = self.linear(self.dropout_out(output))

        if self.residual:
            pred = pred + pred_pre

        return pred, hidden_state


    def forward(self, input, target):
        # check std of previous time
        mask_pred = torch.std(input, dim=0, keepdim=True) > 1e-4
        # Encoder
        input_en = input
        if self.veloc:
            input_vl = torch.zeros(input.size()).to(input.device)
            input_vl[1:] = input[1:] - input[0:-1]
            input_en = torch.cat((input, input_vl), dim=-1)

        if self.pos_embed:
            pos_emb = self.position_embeding[0:input_en.size(0)].unsqueeze(1).repeat(1, input_en.size(1), 1)
            input_en = torch.cat((input_en, pos_emb), dim=-1)

        outputs_enc, hidden_state_en = self.forward_seq(input_en)

        # Decoder
        count = input.size(0) + 1
        outputs_dec = torch.zeros(target.size(0) - 1, target.size(1), target.size(2)).to(input.device)

        for i in range(len(target) - 1):
            inp_cur = target[i][None] if i == 0 else pred
            if self.veloc:
                inp_cur_vl = (target[i][None] - input[-1][None]) if i == 0 else (pred - outputs_dec[i - 1:i])
                inp_cur = torch.cat((inp_cur, inp_cur_vl), dim=-1)

            if self.pos_embed:
                pos_emb = self.position_embeding[count - 1:count].unsqueeze(1).repeat(1, inp_cur.size(1), 1)
                inp_cur = torch.cat((inp_cur, pos_emb), dim=-1)

            pred, hidden_state = self.forward_seq(inp_cur, hidden=(hidden_state_en if i == 0 else hidden_state))

            if self.std_mask:
                pred = mask_pred.float() * pred

            outputs_dec[i:i + 1] = pred
            count += 1
        return outputs_enc, outputs_dec


def position_embedding(d_model, max_len=75): # +25*4
    if d_model <= 0:
        pe = torch.eye(max_len).float()
        pe.require_grad = False
        return pe

    pe = torch.zeros(max_len, d_model).float()
    pe.require_grad = False

    position = torch.arange(0, max_len).float().unsqueeze(1)
    div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super(TransformerModel, self).__init__()
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer_encoder = nn.Transformer(d_model=d_model, nhead=nhead, 
                                                  num_encoder_layers=nlayers, num_decoder_layers=nlayers,
                                                  batch_first=True)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, d_model)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output


if __name__ == '__main__':
    feat = position_embedding(max_len=20, d_model=512)
    print(feat.shape)
    print(feat)