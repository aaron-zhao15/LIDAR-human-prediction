import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import pdb

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class Recurrent_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(Recurrent_GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Define the first 3 layers with GRUs and linear layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.GRU(input_size, hidden_size, batch_first=True))
            self.layers.append(nn.GRU(hidden_size, hidden_size, batch_first=True))
            self.layers.append(nn.Linear(hidden_size, input_size))  # Linear layer

        # Output layer
        self.output_layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Iterate through the layers
        for layer in self.layers:
            if isinstance(layer, nn.GRU):
                x, _ = layer(x)
            else:
                x = layer(x)

        # Final output layer
        output = self.output_layer(x)

        return output


class GRUNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, drop_prob=0.0):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
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
    
class Recurrent_Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Recurrent_Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True, dropout=0.2)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        
        batch_size = x.size(0)
        seq_len = x.size(1)
        joint_dims = x.size(2)//2

        #Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # print(hidden.shape)
        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        out = self.relu(out)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        out = out.reshape((batch_size, seq_len, joint_dims))
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden


