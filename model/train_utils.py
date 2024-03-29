import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from .individual_TF import IndividualTF
from .transformer.batch import subsequent_mask


def train_masks(n_epochs, model, criterion, optimizer, train_loader, validate_loader, test_loader, device):
    epoch_times = []
    epoch_losses = []
    evaluations = []
    for epoch in range(1, n_epochs + 1):
        model = model.train()
        
        start_time = time.perf_counter()
        # h = model.init_hidden(batch_size)
        losses = []
        counter = 0
        for x, label in train_loader:
            x, label = x.to(device).float(), label.to(device).float()
            counter += 1

            # target = label[:, :-1, :]
            target = x[:, :-1, :]
            target_c = torch.ones((target.shape[0], target.shape[1], (target.shape[2]//2)//3)).to(device).float()
            target = torch.cat((target, target_c), -1)
            start_of_seq = torch.zeros((target.shape[0], 1, target.shape[2])).to(device)
            start_of_seq[:, :, -1] = 1

            dec_inp = torch.cat((start_of_seq, target), 1)
            src_att = torch.ones((x.shape[0], 1, x.shape[1])).to(device).float()
            trg_att = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0],1,1).to(device).float()
            
            out = model(x, dec_inp, src_att, trg_att)
            # encoder_out, out = model(x.to(device).float())
            # print(out.shape, label.to(device).float().shape)
            # print(out.shape, label[:,-1:,:].shape)
            loss = criterion(out, label)
            optimizer.optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if counter%200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), np.mean(losses)))
        epoch_losses.append(np.mean(losses))
        evaluation = evaluate_masks(model, validate_loader, criterion, device)
        evaluations.append(evaluation)
        current_time = time.perf_counter()
        if epoch > 0:
            print("Epoch {}/{} Done, Total Loss: {}, Validation Loss: {}".format(epoch, n_epochs, np.mean(losses), evaluation))
            print("Total Time Elapsed: {} seconds".format(str(current_time-start_time)))
        epoch_times.append(current_time-start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    print("Test Loss: {}".format(evaluate_masks(model, test_loader, criterion, device)))
    return epoch_losses, evaluations


def train_pvred(n_epochs, model, criterion, optimizer, train_loader, validate_loader, test_loader, device):
    epoch_times = []
    epoch_losses = []
    evaluations = []
    for epoch in range(1, n_epochs + 1):
        model = model.train()
        
        start_time = time.perf_counter()
        # h = model.init_hidden(batch_size)
        losses = []
        counter = 0
        for x, label in train_loader:
            x, label = x.to(device).float(), label.to(device).float()
            counter += 1

            encoder_out, out = model(x)
            loss = criterion(out, label)
            optimizer.optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if counter%200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), np.mean(losses)))
        epoch_losses.append(np.mean(losses))
        evaluation = evaluate_pvred(model, validate_loader, criterion, device)
        evaluations.append(evaluation)
        current_time = time.perf_counter()
        if epoch > 0:
            print("Epoch {}/{} Done, Total Loss: {}, Validation Loss: {}".format(epoch, n_epochs, np.mean(losses), evaluation))
            print("Total Time Elapsed: {} seconds".format(str(current_time-start_time)))
        epoch_times.append(current_time-start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    print("Test Loss: {}".format(evaluate_pvred(model, test_loader, criterion, device)))
    return epoch_losses, evaluations


def evaluate_masks(model, test_loader, criterion, device):
    # Set the model in evaluation mode (no gradient computation)
    model = model.eval()

    # Initialize a variable to store MSE
    mse_values = []

    # Iterate through the test DataLoader
    with torch.no_grad():
        for x, label in test_loader:
            x, label = x.to(device).float(), label.to(device).float()
            
            # target = label[:, :-1, :]
            target = x[:, :-1, :]
            target_c = torch.ones((target.shape[0], target.shape[1], (target.shape[2]//2)//3)).to(device).float()
            target = torch.cat((target, target_c), -1)
            start_of_seq = torch.zeros((target.shape[0], 1, target.shape[2])).to(device)
            start_of_seq[:, :, -1] = 1

            dec_inp = torch.cat((start_of_seq, target), 1)
            src_att = torch.ones((x.shape[0], 1, x.shape[1])).to(device).float()
            trg_att = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0],1,1).to(device).float()
            
            out = model(x, dec_inp, src_att, trg_att)
            # Forward pass to make predictions using the model
            # encoder_out, out = model(x)
            # Calculate the MSE for the batch
            loss = criterion(out[:,-1:,:], label[:,-1:,:])
            mse = loss.item()
            # Append the MSE value to the list
            mse_values.append(mse)

    # Calculate the overall evaluation metric (average MSE)
    average_mse = np.mean(mse_values)

    return average_mse

def evaluate_pvred(model, test_loader, criterion, device):
    # Set the model in evaluation mode (no gradient computation)
    model = model.eval()

    # Initialize a variable to store MSE
    mse_values = []

    # Iterate through the test DataLoader
    with torch.no_grad():
        for x, label in test_loader:
            x, label = x.to(device).float(), label.to(device).float()
            
            # Forward pass to make predictions using the model
            out, hidden = model(x, label)
            # Calculate the MSE for the batch
            loss = criterion(out, label)
            mse = loss.item()
            # Append the MSE value to the list
            mse_values.append(mse)

    # Calculate the overall evaluation metric (average MSE)
    average_mse = np.mean(mse_values)

    return average_mse

def train_standard(n_epochs, model, criterion, optimizer, train_loader, validate_loader, test_loader, device):
    epoch_times = []
    epoch_losses = []
    evaluations = []
    for epoch in range(1, n_epochs + 1):
        model = model.train()
        
        start_time = time.perf_counter()
        # h = model.init_hidden(batch_size)
        losses = []
        counter = 0
        for x, label in train_loader:
            x, label = x.to(device).float(), label.to(device).float()
            counter += 1
            out, _ = model(x)
            
            # masked label and output comparison
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if counter%200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), np.mean(losses)))
        epoch_losses.append(np.mean(losses))
        evaluation = evaluate_standard(model, validate_loader, criterion, device)
        evaluations.append(evaluation)
        current_time = time.perf_counter()
        if epoch > 0:
            print("Epoch {}/{} Done, Total Loss: {}, Validation Loss: {}".format(epoch, n_epochs, np.mean(losses), evaluation))
            print("Total Time Elapsed: {} seconds".format(str(current_time-start_time)))
        epoch_times.append(current_time-start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    test_loss = evaluate_standard(model, test_loader, criterion, device)
    print("Test Loss: {}".format(test_loss))
    evaluations.append(test_loss)
    return epoch_losses, evaluations

def evaluate_standard(model, test_loader, criterion, device):
    # Set the model in evaluation mode (no gradient computation)
    model = model.eval()

    # Initialize a variable to store MSE
    mse_values = []

    # Iterate through the test DataLoader
    with torch.no_grad():
        for x, label in test_loader:
            x, label = x.to(device).float(), label.to(device).float()
            
            # Forward pass to make predictions using the model
            out, hidden = model(x)
            # Calculate the MSE for the batch
            loss = criterion(out, label)
            mse = loss.item()
            # Append the MSE value to the list
            mse_values.append(mse)

    # Calculate the overall evaluation metric (average MSE)
    average_mse = np.mean(mse_values)

    return average_mse


def train_GT(n_epochs, model, criterion, optimizer, train_loader, validate_loader, test_loader, device):
    epoch_times = []
    epoch_losses = []
    evaluations = []
    for epoch in range(1, n_epochs + 1):
        model = model.train()
        
        start_time = time.perf_counter()
        # h = model.init_hidden(batch_size)
        losses = []
        counter = 0
        for x, label in train_loader:
            x, label = x.to(device).float(), label.to(device).float()
            counter += 1
            out, _ = model(x)
            
            # masked label and output comparison
            label = label[:, -1:, :]
            out = out[:, -1:, :]
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if counter%200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), np.mean(losses)))
        epoch_losses.append(np.mean(losses))
        evaluation = evaluate_GT(model, validate_loader, criterion, device)
        evaluations.append(evaluation)
        current_time = time.perf_counter()
        if epoch > 0:
            print("Epoch {}/{} Done, Total Loss: {}, Validation Loss: {}".format(epoch, n_epochs, np.mean(losses), evaluation))
            print("Total Time Elapsed: {} seconds".format(str(current_time-start_time)))
        epoch_times.append(current_time-start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    test_loss = evaluate_GT(model, test_loader, criterion, device)
    print("Test Loss: {}".format(test_loss))
    evaluations.append(test_loss)
    return epoch_losses, evaluations

def evaluate_GT(model, test_loader, criterion, device):
    # Set the model in evaluation mode (no gradient computation)
    model = model.eval()

    # Initialize a variable to store MSE
    mse_values = []

    # Iterate through the test DataLoader
    with torch.no_grad():
        for x, label in test_loader:
            x, label = x.to(device).float(), label.to(device).float()
            
            # Forward pass to make predictions using the model
            out, hidden = model(x)
            # Calculate the MSE for the batch
            label = label[:, 50:, :]
            out = out[:, 50:, :]
            loss = criterion(out, label)
            mse = loss.item()
            # Append the MSE value to the list
            mse_values.append(mse)

    # Calculate the overall evaluation metric (average MSE)
    average_mse = np.mean(mse_values)

    return average_mse

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, device):

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

def model_to_state_dict(model_filepath="model/trained_model_data/TF_1_small.pt"):
    model = torch.load(model_filepath)
    print("Saving model state dict in ", model_filepath[:-3] + "_statedict.pt")
    torch.save(model.state_dict(), model_filepath[:-3] + "_statedict.pt")
    return

# model_to_state_dict()

