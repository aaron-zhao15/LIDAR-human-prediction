import time
import numpy as np
import logging
import mogaze_utils
from MogazeDataset import MogazeDataset

from sklearn.model_selection import KFold

from models import *

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

joint_dims = 66
seq_len = 4

joint_posns = mogaze_utils.read_from_file("../mogaze_data/p1_1_human_data.hdf5")
joint_vels = mogaze_utils.get_velocities(joint_posns)
joint_posns = joint_posns[:-1]
[input_seqs, target_seqs] = mogaze_utils.sequence_from_array(joint_posns, 4, 3)
[input_vel_seqs, target_vel_seqs] = mogaze_utils.sequence_from_array(joint_vels, 4, 3)

dataset = MogazeDataset(input_seqs, target_seqs, input_vel_seqs, target_vel_seqs)
# print(dataset)

# from dataset, downsample to every 20 frames so dt = 20 frames. then the input seq should be timesteps [i, i+1, i+2]
# starting from timestep i. From an input seq of [i, i+1, i+2], the target_seq should be timesteps [i+2, i+3, i+4].

# input_seq = torch.from_numpy(input_seq)
# target_seq = torch.Tensor(target_seq)

batch_size = 64

# Instantiate the model with hyperparameters
model = Recurrent_Model(input_size=(joint_dims, joint_dims), output_size=joint_dims, hidden_dim=100, n_layers=2)
# We'll also set the model to the device that we defined earlier (default is CPU)
model = model.to(device)

# Define hyperparameters
n_epochs = 100
lr=0.01

# Define Loss, Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

x = (dataset.input_seqs, dataset.input_vels)
y = (dataset.target_seqs, dataset.target_vels)

x_train, x_test, x_validate = torch.utils.data.random_split(x, [0.6, 0.2, 0.2])
y_train, y_test, y_validate = torch.utils.data.random_split(y, [0.6, 0.2, 0.2])

# Implement Dataset and Dataloader in dataset_mogaze.py
train_loader = DataLoader(x_train, batch_size, num_workers=0, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)

# Training Run
input_seq = x_input.to(device)
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    #input_seq = input_seq.to(device)
    output, hidden = model(input_seq.float())
    output = output.to(device)
    target_seq = y_input.to(device)
    loss = criterion(output, target_seq)
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly

    if epoch%10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))
        
        test_output = model(x_test.float())
        print('test loss: ', criterion(test_output, y_test))

