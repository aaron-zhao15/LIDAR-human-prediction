import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch import optim

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
            x, label = x.float(), label.float()
            x, label = x.to(device), label.to(device)
            counter += 1
            out, _ = model(x)
            # masked label and output comparison
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            # if counter%200 == 0:
            #     print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), np.mean(losses)))
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

def train_dual(n_epochs, model, mse_crit, ce_crit, optimizer, train_loader, validate_loader, test_loader, device):
    gamma1 = 1
    gamma2 = 1
    epoch_times = []
    training_losses, training_accuracies = [], []
    validation_losses, validation_accuracies = [], []
    for epoch in range(1, n_epochs + 1):
        model = model.train()
        
        start_time = time.perf_counter()
        # h = model.init_hidden(batch_size)
        losses = []
        counter = 0
        accuracies = torch.zeros(0, device=device)
        for x, label, task in train_loader:
            x, label, task = x.float(), label.float(), task.float()
            x, label, task = x.to(device), label.to(device), task.to(device)
            counter += 1
            decoder_out, encoder_logits = model(x)
            # accuracy calculation
            predictions = torch.argmax(encoder_logits, axis=1)
            true_labels = torch.argmax(task, axis=1)
            accuracy = predictions==true_labels
            accuracies = torch.cat((accuracies, accuracy))

            mse_loss = mse_crit(decoder_out, label)
            ce_loss = ce_crit(encoder_logits, task)
            loss = gamma1*mse_loss + gamma2*ce_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if counter%200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), np.mean(losses)))
        training_losses.append(torch.mean(torch.Tensor(losses)).cpu())
        training_accuracies.append(torch.mean(torch.Tensor(accuracies)).cpu())
        task_accuracy, average_mse = evaluate_dual(model, validate_loader, mse_crit, ce_crit, device)
        task_accuracy, average_mse = task_accuracy.cpu(), average_mse.cpu()
        validation_losses.append(average_mse)
        validation_accuracies.append(task_accuracy)
        current_time = time.perf_counter()
        epoch_accuracy = torch.mean(torch.Tensor(accuracies)).cpu()
        if epoch > 0:
            print("Epoch {}/{} Done, Total Loss: {}, Epoch accuracy: {}, Validation Accuracy: {}".format(epoch, n_epochs, np.mean(losses), epoch_accuracy, task_accuracy))
            print("Validation loss: {}".format(average_mse))
            print("Total Time Elapsed: {} seconds".format(str(current_time-start_time)))
        epoch_times.append(current_time-start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    task_accuracy, average_mse = evaluate_dual(model, test_loader, mse_crit, ce_crit, device)
    task_accuracy, average_mse = task_accuracy.cpu(), average_mse.cpu()
    print("Test Loss: {}, Test Accuracy: {}".format(average_mse, task_accuracy))
    validation_losses.append(average_mse)
    validation_accuracies.append(task_accuracy)
    # [training_losses, training_accuracies], [validation_losses, validation_accuracies]
    return [training_losses, training_accuracies], [validation_losses, validation_accuracies]

def evaluate_dual(model, test_loader, mse_crit, ce_crit, device):
    # Set the model in evaluation mode (no gradient computation)
    model = model.eval()

    # Initialize a variable to store MSE
    accuracies = torch.zeros(0, device=device)
    losses = []

    gamma1 = 1
    gamma2 = 1

    # Iterate through the test DataLoader
    with torch.no_grad():
        for x, label, task in test_loader:
            x, label, task = x.float(), label.float(), task.float()
            x, label, task = x.to(device), label.to(device), task.to(device)
            # Forward pass to make predictions using the model
            decoder_out, encoder_logits = model(x)
            # masked label and output comparison
            # out = out[:, -1, ...]
            predictions = torch.argmax(encoder_logits, axis=1)
            true_labels = torch.argmax(task, axis=1)
            # Calculate accuracy for the batch
            accuracy = predictions==true_labels
            accuracies = torch.cat((accuracies, accuracy))
            
            mse_loss = mse_crit(decoder_out, label)
            ce_loss = ce_crit(encoder_logits, task)
            loss = gamma1*mse_loss + gamma2*ce_loss
            losses.append(loss)
            
    # Calculate the overall evaluation metric (average MSE)
    task_accuracy = torch.mean(torch.Tensor(accuracies))
    average_mse = torch.mean(torch.Tensor(losses))
    return task_accuracy, average_mse

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

def train_classifier(n_epochs, model, criterion, optimizer, train_loader, validate_loader, test_loader, device):
    epoch_times = []
    epoch_losses = []
    evaluations = []
    for epoch in range(1, n_epochs + 1):
        model = model.train()
        start_time = time.perf_counter()
        # h = model.init_hidden(batch_size)
        losses = []
        counter = 0
        accuracies = torch.zeros(0, device=device)
        for x, label in train_loader:
            x, label = x.float(), label.float()
            x, label = x.to(device), label.to(device)
            counter += 1
            out, _ = model(x)
            # out = out[:, -1, ...]
            # accuracy calculation
            predictions = torch.argmax(out, axis=1)
            true_labels = torch.argmax(label, axis=1)
            accuracy = predictions==true_labels
            accuracies = torch.cat((accuracies, accuracy))
            # masked label and output comparison
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if counter%200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), np.mean(losses)))
        epoch_losses.append(np.mean(losses))
        evaluation = evaluate_classifier_accuracy(model, validate_loader, criterion, device)
        evaluations.append(evaluation)
        current_time = time.perf_counter()
        epoch_accuracy = torch.mean(accuracies)
        if epoch > 0:
            print("Epoch {}/{} Done, Total Loss: {}, Epoch accuracy: {}, Validation Accuracy: {}".format(epoch, n_epochs, np.mean(losses), epoch_accuracy, evaluation))
            print("Total Time Elapsed: {} seconds".format(str(current_time-start_time)))
        epoch_times.append(current_time-start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    test_loss = evaluate_classifier_accuracy(model, test_loader, criterion, device)
    print("Test Accuracy: {}".format(test_loss))
    evaluations.append(test_loss)
    return epoch_losses, evaluations

def evaluate_classifier_accuracy(model, test_loader, criterion, device):
    # Set the model in evaluation mode (no gradient computation)
    model = model.eval()

    # Initialize a variable to store MSE
    accuracies = torch.zeros(0, device=device)

    # Iterate through the test DataLoader
    with torch.no_grad():
        for x, label in test_loader:
            x, label = x.float(), label.float()
            x, label = x.to(device), label.to(device)
            # Forward pass to make predictions using the model
            out, hidden = model(x)
            # out = out[:, -1, ...]
            predictions = torch.argmax(out, axis=1)
            true_labels = torch.argmax(label, axis=1)
            # Calculate accuracy for the batch
            accuracy = predictions==true_labels
            accuracies = torch.cat((accuracies, accuracy))
    # Calculate the overall evaluation metric (average MSE)
    total_accuracy = torch.mean(accuracies).cpu()

    return total_accuracy

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

