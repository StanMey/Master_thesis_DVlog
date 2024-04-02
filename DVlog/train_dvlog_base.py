import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# PyTorch TensorBoard support
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from pathlib import Path

from utils.dataloaders import BaseDVlogDataset
from utils.metrics import calculate_performance_measures
from models.model import UnimodalDVlogModel, BimodalDVlogModel


# HARDCODE SOME VARIABLES
# batch size: 32
# epochs: 50
# learning rate: 0.0002
# sequence length (t): 596
SEED = 42
torch.manual_seed(SEED)

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0002
SEQUENCE_LENGTH = 596
DIM_MODEL = 256
N_HEADS_UNIMODAL = 8
N_HEADS_CROSS = 16
USE_GPU = True
USE_STD = False
MODEL_NAME = "bimodal_dvlog_v2_cross16"

# training parameters
modality = "both" # can choose between acoustic, visual, or both
visual_feature_dim = 136
acoustic_feature_dim = 25

# do the checks over the parameters
assert modality in ["visual", "acoustic", "both"], f"Modality type not in choices: {modality}"

# setup the paths
annotations_file = Path(r"./dataset/dvlog_labels_v2.csv")
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
if modality == "acoustic":
    # acoustic unimodel
    model = UnimodalDVlogModel(data_shape=(SEQUENCE_LENGTH, acoustic_feature_dim), d_model=DIM_MODEL, n_heads=N_HEADS_UNIMODAL, use_std=USE_STD)
elif modality == "visual":
    # visual unimodel
    model = UnimodalDVlogModel(data_shape=(SEQUENCE_LENGTH, visual_feature_dim), d_model=DIM_MODEL, n_heads=N_HEADS_UNIMODAL, use_std=USE_STD)
else:
    # bimodal model
    model = BimodalDVlogModel(d_model=DIM_MODEL, n_heads=N_HEADS_CROSS, use_std=USE_STD)

# if torch.cuda.is_available():
#     model.cuda()

# # set the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# run the training
epoch_number = 0
best_vloss = 1_000_000
best_f1 = 0.0

for epoch in range(EPOCHS):  # loop over the dataset multiple times
    print('EPOCH {}:'.format(epoch_number + 1))

    # make sure the gradient is on and do a pass over the data
    model.train(True)
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):

        # get the inputs
        if modality == "acoustic":
            _, inputs, labels = data
        elif modality == "visual":
            inputs, _, labels = data
        else:
            inputs_v, inputs_a, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        if modality == "both":
            outputs = model((inputs_a, inputs_v))
        else:
            outputs = model(inputs)

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
    predictions = []
    y_labels = []
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(val_dataloader):

            # choose the appropriate inputs
            if modality == "acoustic":
                _, v_inputs, vlabels = vdata
            elif modality == "visual":
                v_inputs, _, vlabels = vdata
            else:
                v_inputs_v, v_inputs_a, vlabels = data


            if modality == "both":
                voutputs = model((v_inputs_a, v_inputs_v))
            else:
                voutputs = model(v_inputs)

            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss

            # save the predictions and ground truths from each batch for processing
            predictions.append(voutputs.numpy())
            y_labels.append(vlabels.numpy())
    
    avg_vloss = running_vloss / (i + 1)
    accuracy, _, _, fscore = calculate_performance_measures(y_labels, predictions)
    print('LOSS train {} validation {}'.format(avg_loss, avg_vloss))
    print(f"Validation accuracy: {accuracy}; F1-score: {fscore}")

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'trained_models/model_{}'.format(MODEL_NAME)
        torch.save(model.state_dict(), model_path)
    
    # if fscore > best_f1:
    #     best_f1 = fscore
    #     model_path = 'trained_models/model_{}'.format(MODEL_NAME)
    #     torch.save(model.state_dict(), model_path)
    
    epoch_number += 1


print('Finished Training')
