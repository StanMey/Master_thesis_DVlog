import os
import json
import torch

import torch.nn as nn
import torch.optim as optim

import typer
from typing_extensions import Annotated

# PyTorch TensorBoard support
# from torch.utils.tensorboard import SummaryWriter
from confection import Config
from typing import Union

from torch.utils.data import DataLoader
from pathlib import Path

from models.model import UnimodalDVlogModel, BimodalDVlogModel
from models.trimodal_model import TrimodalDVlogModel

from utils.dataloaders import BiasmitMultimodalEmbeddingsDataset, MultimodalEmbeddingsDataset, SyncedMultimodalEmbeddingsDataset
from utils.metrics import calculate_performance_measures
from utils.util import ConfigDict
from utils.util import validate_config, process_config, set_seed


def train_cli(
    config_path: Path = typer.Argument(..., help="Path to config file", exists=True, allow_dash=True),
    models_path: Annotated[Path, typer.Option(help="Path to saved model folder", exists=True, allow_dash=True)] = Path(r"./trained_models"),
    use_gpu: Annotated[int, typer.Option("--gpu-id", "-g", help="GPU ID or -1 for CPU")] = -1,
    seed: Annotated[int, typer.Option("--seed", "-s", help="The seed used when randomness is introduced")] = 42
):
    """The CLI function handling the interaction part when the user runs the train.py script.
    """
    train(config_path=config_path, output_path=models_path, use_gpu=use_gpu, seed=seed)


def train(
    config_path: Union[str, Path],
    output_path: Union[str, Path],
    use_gpu: int,
    seed: int
):
    """Function to extract and process the configuration file and setup the models and dataloaders.
    """
    # check whether the paths exists
    assert os.path.exists(config_path), "Config file does not exist"
    assert os.path.isdir(output_path), "Output path is not a directory"

    # check the config file on completeness
    config = Config().from_disk(config_path)
    validate_config(config)
    config_dict = process_config(config)

    # begin setting up the model for the training cycle
    #TODO setup the device
    # device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    # set the seed
    set_seed(seed)

    # load in the dataset
    if config_dict.encoder1_use_sync:
        # use the synced multimodal dataloader
        training_data = SyncedMultimodalEmbeddingsDataset("train", config_dict, to_tensor=True)
        val_data = SyncedMultimodalEmbeddingsDataset("val", config_dict, to_tensor=True)
    
    elif config_dict.bias_mit:
        # we will use a bias mitigation approach
        training_data = BiasmitMultimodalEmbeddingsDataset("train", config_dict, to_tensor=True, seed=seed)
        val_data = MultimodalEmbeddingsDataset("val", config_dict, to_tensor=True)

    else:
        # use the regular multimodal dataloader
        training_data = MultimodalEmbeddingsDataset("train", config_dict, to_tensor=True)
        val_data = MultimodalEmbeddingsDataset("val", config_dict, to_tensor=True)

    # setup the dataloader
    train_dataloader = DataLoader(training_data, batch_size=config_dict.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=config_dict.batch_size, shuffle=True)

    # setup the model
    if config_dict.n_modalities == 1:
        # initiate an unimodal model
        model = UnimodalDVlogModel((config_dict.sequence_length, config_dict.encoder1_dim),
                                   d_model=config_dict.dim_model, n_heads=config_dict.uni_n_heads, use_std=config_dict.detectlayer_use_std)

    elif config_dict.n_modalities == 2:
        # initiate a bimodal model
        model = BimodalDVlogModel((config_dict.sequence_length, config_dict.encoder1_dim), (config_dict.sequence_length, config_dict.encoder2_dim), cross_type=config_dict.multi_bi_type,
                                   d_model=config_dict.dim_model, uni_n_heads=config_dict.uni_n_heads, cross_n_heads=config_dict.multi_n_heads, use_std=config_dict.detectlayer_use_std)

    else:
        # initiate a trimodal model
        seq_length = config_dict.sequence_length
        model = TrimodalDVlogModel((seq_length, config_dict.encoder1_dim), (seq_length, config_dict.encoder2_dim), (seq_length, config_dict.encoder3_dim), cross_type=config_dict.multi_tri_type,
                                   d_model=config_dict.dim_model, uni_n_heads=config_dict.uni_n_heads, cross_n_heads=config_dict.multi_n_heads, use_std=config_dict.detectlayer_use_std)

    # train the actual model
    train_model(model, train_dataloader, val_dataloader, config_dict, output_path)


def train_model(model, train_dataloader: DataLoader, val_dataloader: DataLoader, config_dict: ConfigDict, output_path: Path):
    """Run the training process using the model and dataloader.
    """
    # setup the saving directory
    model_output_path = os.path.join(output_path, config_dict.model_name)
    os.makedirs(model_output_path, exist_ok=True)

    # set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config_dict.learning_rate)

    # run the training
    best_vloss = 1_000_000

    for epoch in range(config_dict.epochs):  # loop over the dataset multiple times
        print('EPOCH {}:'.format(epoch + 1))

        # make sure the gradient is on and do a pass over the data
        model.train(True)
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):

            # get the inputs
            if config_dict.n_modalities == 1:
                # only one modality so unpack the set
                inputs, labels = data
            else:
                # more modalities so keep the tuple set
                inputs, labels = data[:-1], data[-1]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
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

                # get the validation inputs
                if config_dict.n_modalities == 1:
                    # only one modality so unpack the set
                    v_inputs, vlabels = vdata
                else:
                    # more modalities so keep the input tuple set
                    v_inputs, vlabels = vdata[:-1], vdata[-1]

                # get the predictions of the model
                voutputs = model(v_inputs)

                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss

                # save the predictions and ground truths from each batch for processing
                predictions.append(voutputs.numpy())
                y_labels.append(vlabels.numpy())
        
        avg_vloss = running_vloss / (i + 1)
        accuracy, _, _, fscore, _ = calculate_performance_measures(y_labels, predictions)
        print('LOSS train {} validation {}'.format(avg_loss, avg_vloss))
        print(f"Validation accuracy: {accuracy}; F1-score: {fscore}")

        # Track best performance, and save the model's state (for Inference later on)
        if avg_vloss < best_vloss:
            # save the model itself
            best_vloss = avg_vloss
            model_path = os.path.join(model_output_path, f"model_{config_dict.model_name}.pth")
            torch.save(model.state_dict(), model_path)

    # save the config dictionary
    with open(os.path.join(model_output_path, "model_params.json"), "w") as outfile: 
        json.dump(config_dict.to_dict(), outfile)

    print('Finished Training')


if __name__ == "__main__":
    typer.run(train_cli)
