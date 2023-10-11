import time
import numpy as np
import logging
import mogaze_utils
from MogazeDataset import MogazeDataset
from PVRNN.enc_dec import Encoder_Decoder 
from PVRNN.batch_sample import generate_train_data

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

# joint_dims = 66
joint_dims = 2
seq_len = 5
target_offset = 3
step_size = 1

# joint_posns = mogaze_utils.read_from_hdf("../mogaze_data/p1_1_human_data.hdf5")
joint_posns = mogaze_utils.read_from_csv("../low_dim_data/angles5.txt")
joint_posns = mogaze_utils.downsample_data(joint_posns)
joint_vels = mogaze_utils.get_velocities(joint_posns, dt=0.01)
# joint_posns, (pos_mean, pos_std) = mogaze_utils.normalize(joint_posns)
# joint_vels, (vels_mean, vels_std) = mogaze_utils.normalize(joint_vels)
joint_posns = joint_posns[:-1]
[input_seqs, target_seqs] = mogaze_utils.sequence_from_array(joint_posns, seq_len, target_offset, step_size)
[input_vel_seqs, target_vel_seqs] = mogaze_utils.sequence_from_array(joint_vels, seq_len, target_offset, step_size)

dataset = MogazeDataset(input_seqs, target_seqs, input_vel_seqs, target_vel_seqs)
# print(dataset)


# from dataset, downsample to every 20 frames so dt = 20 frames. then the input seq should be timesteps [i, i+1, i+2]
# starting from timestep i. From an input seq of [i, i+1, i+2], the target_seq should be timesteps [i+2, i+3, i+4].

# input_seq = torch.from_numpy(input_seq)
# target_seq = torch.Tensor(target_seq)

batch_size = 64

# joint_posns_tst = mogaze_utils.read_from_csv("../low_dim_data/angles3.txt")
# joint_posns_tst = mogaze_utils.downsample_data(joint_posns_tst)
# joint_vels_tst = mogaze_utils.get_velocities(joint_posns_tst, dt=0.01)
# joint_posns_tst = joint_posns_tst[:-1]
# [input_seqs_tst, target_seqs_tst] = mogaze_utils.sequence_from_array(joint_posns_tst, seq_len, target_offset, step_size)
# [input_vel_seqs_tst, target_vel_seqs_tst] = mogaze_utils.sequence_from_array(joint_vels_tst, seq_len, target_offset, step_size)
# test_dataset = MogazeDataset(input_seqs_tst, target_seqs_tst, input_vel_seqs_tst, target_vel_seqs_tst)
# test_separate_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=True)

# Instantiate the model with hyperparameters
model = Recurrent_Model(input_size=joint_dims*2, output_size=joint_dims*2, hidden_dim=100, n_layers=2)
# model = Encoder_Decoder(input_size=joint_dims, hidden_size=100, num_layer=2, rnn_unit='gru', veloc=True)
# We'll also set the model to the device that we defined earlier (default is CPU)
model = model.to(device)

# Define hyperparameters
n_epochs = 1500
lr=0.01

# Define Loss, Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train, validate, test = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2])

# Implement Dataset and Dataloader in dataset_mogaze.py
train_loader = DataLoader(train, batch_size=batch_size, num_workers=0, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, num_workers=0, shuffle=True)
validate_loader = DataLoader(validate, batch_size=batch_size, num_workers=0, shuffle=True)

epoch_times = []
for epoch in range(1, n_epochs + 1):
    start_time = time.perf_counter()
    # h = model.init_hidden(batch_size)
    losses = []
    counter = 0
    for x, label in train_loader:
        counter += 1
        model.zero_grad()
        
        # out, h = model(x.to(device).float(), label.to(device).float())
        out, h = model(x.to(device).float())
        loss = criterion(out, label.to(device).float())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if counter%200 == 0:
            print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), avg_loss/counter))
    current_time = time.perf_counter()
    if epoch % 100 ==0 and epoch > 0:
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, n_epochs, np.mean(losses)))
        print("Total Time Elapsed: {} seconds".format(str(current_time-start_time)))
    epoch_times.append(current_time-start_time)
print("Total Training Time: {} seconds".format(str(sum(epoch_times))))


def evaluate(model, test_loader):
    # Set the model in evaluation mode (no gradient computation)
    model.eval()

    # Initialize a variable to store MSE
    mse_values = []

    # Iterate through the test DataLoader
    with torch.no_grad():
        for data, labels in test_loader:
            # Forward pass to make predictions using the model
            predictions, h = model(data.to(device).float())

            # Calculate the MSE for the batch
            loss = criterion(predictions, labels.to(device).float())
            mse = loss.item()

            print(data.shape)
            # Append the MSE value to the list
            mse_values.append(mse)

    # Calculate the overall evaluation metric (average MSE)
    average_mse = np.mean(mse_values)

    return average_mse

print(evaluate(model, test_loader))
# print(evaluate(model, test_separate_loader))


