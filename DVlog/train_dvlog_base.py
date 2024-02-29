import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from pathlib import Path

from utils.dataloaders import BaseDVlogDataset
from models.unimodel import UnimodalDVlogModel


# HARDCODE SOME VARIABLES
# batch size: 32
# epochs: 50
# learning rate: 0.0002
# sequence length (t): 596
EPOCHS = 1
BATCH_SIZE = 1
LEARNING_RATE = 0.0002
SEQUENCE_LENGTH = 596


# load in the dataset
annotations_file = Path(r"./dataset/dvlog_labels.csv")
data_dir = Path(r"./dataset/dvlog-dataset")

training_data = BaseDVlogDataset(annotations_file, data_dir, "train", sequence_length=SEQUENCE_LENGTH)
val_data = BaseDVlogDataset(annotations_file, data_dir, "val", sequence_length=SEQUENCE_LENGTH)

# setup the dataloader
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

# setup the network
model = UnimodalDVlogModel(25, 5)

# # set the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# run the training
for epoch in range(EPOCHS):  # loop over the dataset multiple times
    print('EPOCH {}:'.format(epoch + 1))

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

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 20 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

print('Finished Training')
