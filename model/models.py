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
    

class EncoderDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, rnn_unit, residual=False, out_dropout=0, std_mask=False,
                 veloc=True, pos_embed=True, pos_embed_dim=96, device="cuda"):
        super(EncoderDecoder, self).__init__()
        self.dropout_out = nn.Dropout(p=out_dropout).to(device)
        self.linear = nn.Linear(hidden_size, input_size).to(device)
        self.position_embeding = position_embedding(d_model=pos_embed_dim)
        
        rnn_input = 2*input_size if veloc else input_size
        self.pos_embed = pos_embed
        if pos_embed:
            self.position_embeding = position_embedding(d_model=pos_embed_dim)
            rnn_input = rnn_input + self.position_embeding.size(1)
            self.position_embeding = self.position_embeding.to(device)
        self.rnn = nn.GRU(rnn_input, hidden_size, num_layers=num_layer).to(device)
        
        self.encoder = Encoder(input_size, self.rnn, self.dropout_out, self.linear, self.position_embeding,
                               residual, veloc, pos_embed, device)
        self.decoder = Decoder(input_size, self.rnn, self.dropout_out, self.linear, self.position_embeding,
                               residual, std_mask, veloc, pos_embed, device)
        
    def forward(self, input):
        # check std of previous time
        mask_pred = torch.std(input, dim=0, keepdim=True) > 1e-4
        # Encoder
        outputs_enc, hidden_state_en = self.encoder(input)

        # Decoder
        outputs_dec = self.decoder(input, hidden_state_en, mask_pred)

        return outputs_enc, outputs_dec


class Encoder(nn.Module):
    def __init__(self, input_size, rnn_unit, dropout_out, linear, embedding, residual=False, 
                 veloc=True, pos_embed=True, device="cuda"):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.residual = residual
        self.rnn = rnn_unit
        self.dropout_out = dropout_out
        self.linear = linear
        self.veloc = veloc
        self.pos_embed = pos_embed
        self.position_embeding = embedding

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

    def forward(self, input):
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
        return outputs_enc, hidden_state_en
    
class Decoder(nn.Module):
    def __init__(self, input_size, rnn_unit, dropout_out, linear, embedding, residual=False, std_mask=False,
                 veloc=True, pos_embed=True, device="cuda"):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.residual = residual
        self.rnn = rnn_unit
        self.dropout_out = dropout_out
        self.linear = linear
        self.std_mask = std_mask
        self.veloc = veloc
        self.pos_embed = pos_embed
        self.position_embeding = embedding

    def forward_seq(self, input, hidden=None):
        pred_pre = input[:,:,0:self.input_size].clone()
        if hidden is None:
            output, hidden_state = self.rnn(input)
        else:
            output, hidden_state = self.rnn(input, hidden)
        pred = self.linear(self.dropout_out(output))

        if self.residual:
            pred = pred + pred_pre

        return pred, hidden_state

    def forward(self, input, hidden_state_en, mask_pred):
        # Decoder
        count = input.size(0) + 1
        outputs_dec = torch.zeros(input.size(0), input.size(1), input.size(2)).to(input.device)

        for i in range(len(input)):
            inp_cur = input[i][None] if i == 0 else pred
            if self.veloc:
                inp_cur_vl = (input[i][None] - input[-1][None]) if i == 0 else (pred - outputs_dec[i - 1:i])
                inp_cur = torch.cat((inp_cur, inp_cur_vl), dim=-1)

            if self.pos_embed:
                pos_emb = self.position_embeding[count - 1:count].unsqueeze(1).repeat(1, inp_cur.size(1), 1)
                
                inp_cur = torch.cat((inp_cur, pos_emb), dim=-1)

            pred, hidden_state = self.forward_seq(inp_cur, hidden=(hidden_state_en if i == 0 else hidden_state))

            if self.std_mask:
                pred = mask_pred.float() * pred

            outputs_dec[i:i + 1] = pred
            count += 1
        return outputs_dec
    
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
    
    def forward(self, x):
        
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

class Self_Attention_Encoder_Decoder(nn.Module):
    def __init__(self, input_size, rnn_unit, dropout_out, linear, embedding, residual=False, 
                 veloc=True, pos_embed=True, device="cuda"):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.residual = residual
        self.rnn = rnn_unit
        self.dropout_out = dropout_out
        self.linear = linear
        self.veloc = veloc
        self.pos_embed = pos_embed
        self.position_embeding = embedding

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



def position_embedding(d_model, max_len=150): # +25*4
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
    def __init__(self, input_size: int, output_size: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        super().__init__()
        self.model_type = 'Transformer'
        # self.embedder = MLP(num_layers=nlayers, hidden_size=d_hid, dropout_probability=dropout, input_features=input_size, output_size=d_hid)
        self.pe = PositionalEncoding(d_model=input_size, dropout=dropout)
        encoding_layer = nn.TransformerEncoderLayer(input_size, nhead, dim_feedforward=d_hid, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoding_layer, nlayers)
        decoding_layer = nn.TransformerDecoderLayer(input_size, nhead, dim_feedforward=d_hid, dropout=dropout, batch_first=False)
        self.decoder = nn.TransformerDecoder(decoding_layer, nlayers)
        self.d_model = input_size
        self.linear = nn.Linear(input_size, output_size)

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
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(src.shape[1]).to(device)
        positional_embedding = self.pe(src)
        encoded = self.encoder(positional_embedding, src_mask)
        decoded = self.decoder(src, encoded, src_mask)
        output = self.linear(decoded)
        return output

class ShallowTransformer(nn.Module):
    '''
    Encoder-only transformer-based model
    Takes input of shape (batch_size, sequence_length, num_joints * features_per_joint)
    '''
    def __init__(self, embedding_dim, embedding_hidden_size, embedding_num_layers, num_stacks, num_heads, 
                 transformer_mlp_size, dropout_probability = 0.1,  
                 output_size = 1, num_joints=9, feats_per_joint=3, num_timesteps=20):
        super(ShallowTransformer, self).__init__()
        self.embedder = MLP(embedding_num_layers, embedding_hidden_size, embedding_dim, dropout_probability=dropout_probability, input_features = num_joints * feats_per_joint)
        self.positional = PositionalEncoding(embedding_dim, dropout_probability, num_timesteps)
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_heads, transformer_mlp_size, dropout_probability, batch_first = True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_stacks)
        self.fc = nn.Linear(embedding_dim * num_timesteps, output_size)
    
    def forward(self, x):
        x = self.embedder(x)
        x = self.positional(x)
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
    
class MLP(nn.Module):
    '''
    Standard MLP. Takes flattened inputs of shape (batch_size, sequence_length * num_joints * features_per_joint)
    '''
    def __init__(self, num_layers=1, hidden_size = 100, output_size = 1, dropout_probability=0.2,
                 input_features = 540):
        super(MLP, self).__init__()

        self.fc1 = nn.Sequential(
                nn.Linear(input_features, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_probability)
            )
        # Create MLP layers
        self.mlp_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_probability)
            )
            for _ in range(num_layers - 1)
        ])
        self.last = nn.Linear(hidden_size, output_size)  # Adjust the output size based on your task

    def forward(self, x):
        x = self.fc1(x)
        for mlp_layer in self.mlp_layers:
            x = mlp_layer(x)
        x = self.last(x)
        return x


if __name__ == '__main__':
    feat = position_embedding(max_len=20, d_model=512)
    print(feat.shape)
    print(feat)