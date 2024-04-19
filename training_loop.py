import time
import numpy as np
import logging
from model import data_utils, train_utils
from model.datasets import TrajectoryDataset

from model.models import *
from model.individual_TF import IndividualTF
from model.decoder_GT import Decoder_GPT
from model.encoder_GT import Encoder_GPT, Encoder_GPT_classifier, BiLSTM, Transformer
from model.encoder_decoder_GT import Encoder_Decoder_GPT, Encoder_Decoder_Classifier

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW

from model.transformer.noam_opt import NoamOpt

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

joint_dims = 66
seq_len = 200
target_offset = 60
step_size = 10
hidden_size = 1024
num_classes = 17
# hidden_size = 64

# joint_posns = mogaze_utils.read_from_hdf("../mogaze_data/p1_1_human_data.hdf5")
# joint_posns = mogaze_utils.downsample_data(joint_posns)
# joint_vels = mogaze_utils.get_velocities(joint_posns, dt=0.01)
# joint_posns, (pos_mean, pos_std) = mogaze_utils.normalize(joint_posns)
# joint_vels, (vels_mean, vels_std) = mogaze_utils.normalize(joint_vels)
# joint_posns = joint_posns[:-1]
# [input_seqs, target_seqs] = mogaze_utils.sequence_from_array(joint_posns, seq_len, target_offset, step_size)
# [input_vel_seqs, target_vel_seqs] = mogaze_utils.sequence_from_array(joint_vels, seq_len, target_offset, step_size)

# dataset = MogazeDataset(input_seqs, target_seqs, input_vel_seqs, target_vel_seqs)

# dataset = data_utils.generate_data_from_csv_folder("../low_dim_data/", seq_len, target_offset, step_size)
# dataset = data_utils.generate_data_from_hdf_folder("humoro/mogaze/", seq_len, target_offset, step_size, use_vel=False)
# dataset = data_utils.generate_data_from_hdf_file("humoro/mogaze/p2_1_human_data.hdf5", seq_len, target_offset, step_size, use_vel=False)
# dataset = data_utils.generate_GT_data_from_hdf_file("humoro/mogaze/p1_1_human_data.hdf5", seq_len, target_offset, step_size, use_vel=False)
# dataset = data_utils.generate_GT_data_from_hdf_folder("humoro/mogaze/", seq_len, target_offset, step_size)
# dataset = data_utils.generate_intent_data_from_person("humoro/mogaze/p2_1", step_size=step_size, sample_len=seq_len, offset_len=target_offset, use_vel=False)
dataset = data_utils.generate_intent_segments_from_folder("humoro/mogaze/", seq_len=seq_len, step_size=step_size)
# print(dataset)


# from dataset, downsample to every 20 frames so dt = 20 frames. then the input seq should be timesteps [i, i+1, i+2]
# starting from timestep i. From an input seq of [i, i+1, i+2], the target_seq should be timesteps [i+2, i+3, i+4].

# input_seq = torch.from_numpy(input_seq)
# target_seq = torch.Tensor(target_seq)

batch_size = 64
# print(device)
# Instantiate the model with hyperparameters
# model = RNN_model(input_size=joint_dims*2, output_size=joint_dims*2, hidden_dim=hidden_size, n_layers=2)
# model = Encoder_Decoder(input_size=joint_dims*2, hidden_size=hidden_size, num_layer=32, rnn_unit='gru', veloc=False, device=device)
# model = TransformerModel(joint_dims*2, joint_dims*2, 1, hidden_size, 16, 0.1).to(device)
# model = EncoderDecoder(input_size=joint_dims*2, hidden_size=hidden_size, num_layer=32, rnn_unit='gru', veloc=False, device=device)
# model = IndividualTF(enc_inp_size=joint_dims*2, dec_inp_size=(joint_dims*2)+(joint_dims//3), dec_out_size=joint_dims*2, device=device)

# block_size should be either seq_len or seq_len*2-1, depending on the dataset format
# model = Decoder_GPT(n_layer=6, n_head=6, n_embd=192, vocab_size=joint_dims, block_size=seq_len, pdrop=0.1, device=device)
# model = Encoder_GPT_classifier(n_layer=6, n_head=6, n_embd=192, vocab_size=66, block_size=seq_len, num_classes=num_classes, pdrop=0.1, device=device)
model = Encoder_Decoder_GPT(n_layer=3, n_head=6, n_embd=192, vocab_size=joint_dims, block_size=seq_len, pdrop=0.1, device=device)
# model = Encoder_Decoder_Classifier(n_layer=3, n_head=6, n_embd=192, vocab_size=129, block_size=seq_len, num_classes=num_classes, pdrop=0.1, device=device)
# model = torch.load('TransformerModel4.pt')
# model.load_state_dict(torch.load('model/trained_model_data/GT_1_small_statedict.pt'))
# config = {'input_dim': 66,
#             'num_layers': 6,
#             'lstm_hidden': 1024,
#             'lstm_dropout': 0.1,
#             'fc_dim': 1024,
#             'num_classes': 17}
# config = {'input_dim': 66,
#                   'model_name': 'bert',
#                   'config_name': 'bert',
#                   'config_dict': dict(num_hidden_layers=6),
#                   'use_pretrained': True,
#                   'max_video_len': seq_len,
#                   'fc_dim': 1024,
#                   'num_classes': 17}
# model = Transformer(config, device)


# Define hyperparameters
n_epochs = 1000
lr=1e-3

# Define Loss, Optimizer
# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()

train, validate, test = torch.utils.data.random_split(dataset, [0.6, 0.2, 0.2])

# Implement Dataset and Dataloader in dataset_mogaze.py
train_loader = DataLoader(train, batch_size=batch_size, num_workers=0, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, num_workers=0, shuffle=True)
validate_loader = DataLoader(validate, batch_size=batch_size, num_workers=0, shuffle=True)

# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = NoamOpt(512, 1, len(train_loader)*10, torch.optim.Adam(model.parameters(), lr=lr))

epoch_losses, evaluations = train_utils.train_standard(n_epochs, model, criterion, optimizer, train_loader, validate_loader, test_loader, device)
# epoch_losses, evaluations = train_utils.train_classifier(n_epochs, model, criterion, optimizer, train_loader, validate_loader, test_loader, device)
# epoch_losses, evaluations = train_utils.train_pvred(n_epochs, model, criterion, optimizer, train_loader, validate_loader, test_loader, device)

np.savetxt('model/trained_model_data/epoch_losses_E_GT_C.gz', epoch_losses)
np.savetxt('model/trained_model_data/evaluations_E_GT_C.gz', evaluations)
torch.save(model.state_dict(), 'model/trained_model_data/E_GT_C_statedict.pt')


