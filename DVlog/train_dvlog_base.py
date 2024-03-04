import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# PyTorch TensorBoard support
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from pathlib import Path

from utils.dataloaders import BaseDVlogDataset
from models.unimodel import UnimodalDVlogModel


# HARDCODE SOME VARIABLES
# batch size: 32
# epochs: 50
# learning rate: 0.0002
# sequence length (t): 596
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0002
SEQUENCE_LENGTH = 596
USE_GPU = True
MODEL_NAME = "unimodal_acoustic_v1"


# setup the paths
annotations_file = Path(r"./dataset/dvlog_labels_v1.csv")
data_dir = Path(r"./dataset/dvlog-dataset")

# setup the device
# device = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")

# load in the dataset
training_data = BaseDVlogDataset(annotations_file, data_dir, "train", sequence_length=SEQUENCE_LENGTH, to_tensor=True)
val_data = BaseDVlogDataset(annotations_file, data_dir, "val", sequence_length=SEQUENCE_LENGTH, to_tensor=True)

# setup the dataloader
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

# setup the network
model = UnimodalDVlogModel(25, 5)

# if torch.cuda.is_available():
#     model.cuda()

# # set the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# run the training
epoch_number = 0
best_vloss = 1_000_000

for epoch in range(EPOCHS):  # loop over the dataset multiple times
    print('EPOCH {}:'.format(epoch_number + 1))

    # make sure the gradient is on and do a pass over the data
    model.train(True)
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):

        # get the inputs
        _, acoustic_inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(acoustic_inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # save statistics
        running_loss += loss.item()
    avg_loss = running_loss / (i + 1)
    
    # Per-Epoch Activity
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    running_vloss = 0.0
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(val_dataloader):
            _, v_acoustic_inputs, vlabels = vdata

            voutputs = model(v_acoustic_inputs)
            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss
    
    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}'.format(MODEL_NAME)
        torch.save(model.state_dict(), model_path)
    
    epoch_number += 1


print('Finished Training')
