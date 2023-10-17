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
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW

def standard_train(n_epochs, model, criterion, optimizer, train_loader, validate_loader, test_loader):
    epoch_times = []
    for epoch in range(1, n_epochs + 1):
        start_time = time.perf_counter()
        # h = model.init_hidden(batch_size)
        losses = []
        counter = 0
        for x, label in train_loader:
            counter += 1
            model.zero_grad()
            
            out, h = model(x.to(device).float(), label.to(device).float())
            # out, h = model(x.to(device).float(), label)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if counter%200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), np.mean(losses)))
        current_time = time.perf_counter()
        if epoch > 0:
            print("Epoch {}/{} Done, Total Loss: {}, Validation Loss: {}".format(epoch, n_epochs, np.mean(losses), evaluate(model, validate_loader, criterion)))
            print("Total Time Elapsed: {} seconds".format(str(current_time-start_time)))
        epoch_times.append(current_time-start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))

    print("Test Loss: {}".format(evaluate(model, test_loader, criterion)))


def evaluate(model, test_loader, criterion):
    # Set the model in evaluation mode (no gradient computation)
    model = model.eval()

    # Initialize a variable to store MSE
    mse_values = []

    # Iterate through the test DataLoader
    with torch.no_grad():
        for x, label in test_loader:
            # Forward pass to make predictions using the model
            predictions, h = model(x.to(device).float(), label.to(device).float())
            # Calculate the MSE for the batch
            loss = criterion(predictions, label.to(device).float())
            mse = loss.item()
            # Append the MSE value to the list
            mse_values.append(mse)

    # Calculate the overall evaluation metric (average MSE)
    average_mse = np.mean(mse_values)

    return average_mse

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))