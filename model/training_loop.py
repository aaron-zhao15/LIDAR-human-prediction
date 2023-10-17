import time
import numpy as np
import logging
import mogaze_utils
from TrajectoryDataset import TrajectoryDataset
from PVRNN.enc_dec import Encoder_Decoder 
from PVRNN.batch_sample import generate_train_data

from models import *

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW

import train_utils

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

joint_dims = 66
# joint_dims = 2
seq_len = 10
target_offset = 3
step_size = 1
hidden_size = 128

# joint_posns = mogaze_utils.read_from_hdf("../mogaze_data/p1_1_human_data.hdf5")
# joint_posns = mogaze_utils.downsample_data(joint_posns)
# joint_vels = mogaze_utils.get_velocities(joint_posns, dt=0.01)
# joint_posns, (pos_mean, pos_std) = mogaze_utils.normalize(joint_posns)
# joint_vels, (vels_mean, vels_std) = mogaze_utils.normalize(joint_vels)
# joint_posns = joint_posns[:-1]
# [input_seqs, target_seqs] = mogaze_utils.sequence_from_array(joint_posns, seq_len, target_offset, step_size)
# [input_vel_seqs, target_vel_seqs] = mogaze_utils.sequence_from_array(joint_vels, seq_len, target_offset, step_size)

# dataset = MogazeDataset(input_seqs, target_seqs, input_vel_seqs, target_vel_seqs)

# dataset = mogaze_utils.generate_data_from_csv_folder("../low_dim_data/", seq_len, target_offset, step_size)
dataset = mogaze_utils.generate_data_from_hdf_folder("../mogaze_data/", seq_len, target_offset, step_size)

# print(dataset)


# from dataset, downsample to every 20 frames so dt = 20 frames. then the input seq should be timesteps [i, i+1, i+2]
# starting from timestep i. From an input seq of [i, i+1, i+2], the target_seq should be timesteps [i+2, i+3, i+4].

# input_seq = torch.from_numpy(input_seq)
# target_seq = torch.Tensor(target_seq)

batch_size = 64

# Instantiate the model with hyperparameters
# model = RNN_model(input_size=joint_dims*2, output_size=joint_dims*2, hidden_dim=hidden_size, n_layers=2)
model = Encoder_Decoder(input_size=joint_dims*2, hidden_size=hidden_size, num_layer=2, rnn_unit='gru', veloc=False)
# encoder = EncoderRNN(input_size=joint_dims*2, hidden_size=hidden_size, seq_len=seq_len).to(device)
# decoder = DecoderRNN(hidden_size=hidden_size, output_size=joint_dims*2, seq_len=seq_len)

# Define hyperparameters
n_epochs = 400
lr=0.001

# Define Loss, Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train, validate, test = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2])

# Implement Dataset and Dataloader in dataset_mogaze.py
train_loader = DataLoader(train, batch_size=batch_size, num_workers=0, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, num_workers=0, shuffle=True)
validate_loader = DataLoader(validate, batch_size=batch_size, num_workers=0, shuffle=True)

# train_utils.train(train_loader, encoder, decoder, n_epochs, learning_rate=lr)
train_utils.standard_train(n_epochs, model, criterion, optimizer, train_loader, validate_loader, test_loader)

