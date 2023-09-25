import time
import numpy as np
import logging
import mogaze_utils
from MogazeDataset import MogazeDataset

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
model = GRUNet(input_size=joint_dims*2, output_size=joint_dims, hidden_dim=100, n_layers=2)
# We'll also set the model to the device that we defined earlier (default is CPU)
model = model.to(device)

# Define hyperparameters
n_epochs = 10
lr=0.1

# Define Loss, Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train, test, validate = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2])

# Implement Dataset and Dataloader in dataset_mogaze.py
train_loader = DataLoader(train, batch_size=batch_size, num_workers=0, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, num_workers=0, shuffle=True)
validate_loader = DataLoader(validate, batch_size=batch_size, num_workers=0, shuffle=True)

epoch_times = []
for epoch in range(1, n_epochs + 1):
    start_time = time.perf_counter()
    # h = model.init_hidden(batch_size)
    avg_loss = 0.
    counter = 0
    for x, label in train_loader:
        counter += 1
        model.zero_grad()
        
        out, h = model(x.to(device).float())
        loss = criterion(out, label.to(device).float())
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        if counter%200 == 0:
            print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), avg_loss/counter))
    current_time = time.perf_counter()
    print("Epoch {}/{} Done, Total Loss: {}".format(epoch, n_epochs, avg_loss/len(train_loader)))
    print("Total Time Elapsed: {} seconds".format(str(current_time-start_time)))
    epoch_times.append(current_time-start_time)
print("Total Training Time: {} seconds".format(str(sum(epoch_times))))




def evaluate(model, test_x, test_y, label_scalers):
    model.eval()
    outputs = []
    targets = []
    start_time = time.clock()
    for i in test_x.keys():
        inp = torch.from_numpy(np.array(test_x[i]))
        labs = torch.from_numpy(np.array(test_y[i]))
        h = model.init_hidden(inp.shape[0])
        out, h = model(inp.to(device).float(), h)
        outputs.append(label_scalers[i].inverse_transform(out.cpu().detach().numpy()).reshape(-1))
        targets.append(label_scalers[i].inverse_transform(labs.numpy()).reshape(-1))
    print("Evaluation Time: {}".format(str(time.clock()-start_time)))
    sMAPE = 0
    for i in range(len(outputs)):
        sMAPE += np.mean(abs(outputs[i]-targets[i])/(targets[i]+outputs[i])/2)/len(outputs)
    print("sMAPE: {}%".format(sMAPE*100))
    return outputs, targets, sMAPE

