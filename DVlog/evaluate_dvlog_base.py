import torch
import typer

# PyTorch TensorBoard support
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from pathlib import Path

from utils.dataloaders import BaseDVlogDataset
from utils.metrics import calculate_performance_measures, calculate_fairness_measures
from models.unimodel import UnimodalDVlogModel


# HARDCODE SOME VARIABLES
# batch size: 32
# epochs: 50
# learning rate: 0.0002
# sequence length (t): 596
BATCH_SIZE = 32
SEQUENCE_LENGTH = 596
USE_GPU = True
SAVED_MODEL_WEIGHTS = "unimodal_acoustic_v2"
SAVED_MODEL_PATH = Path(f"trained_models/model_{SAVED_MODEL_WEIGHTS}")

# evaluation parameters
modality = "acoustic" # can choose between acoustic or visual
dataset = "test" # can choose between 'test', 'train', or 'val'
input_dimension = 25
attention_heads = 5

# do the checks over the parameters
assert modality in ["visual", "acoustic"], f"Modality type not in choices: {modality}"
assert dataset in ["test", "train", "val"], f"Chosen dataset not in choices: {dataset}"
assert SAVED_MODEL_PATH.is_file(), f"Saved model not found: {SAVED_MODEL_PATH}"


# setup the paths
annotations_file = Path(r"./dataset/dvlog_labels_v2.csv")
data_dir = Path(r"./dataset/dvlog-dataset")


# load the saved version of the model
saved_model = UnimodalDVlogModel(input_dimension, attention_heads)
saved_model.load_state_dict(torch.load(SAVED_MODEL_PATH))
# Set the model to evaluation mode
saved_model.eval()

# setup the device
# device = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")

# load in the dataset
eval_dataset = BaseDVlogDataset(annotations_file, data_dir, dataset=dataset, sequence_length=SEQUENCE_LENGTH, to_tensor=True, with_protected=True)

# setup the dataloader
eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)

# if torch.cuda.is_available():
#     model.cuda()

predictions = []
y_labels = []
protected = []

# Disable gradient computation and reduce memory consumption.
with torch.no_grad():
    for i, vdata in enumerate(eval_dataloader):

        # choose the appropriate inputs
        if modality == "acoustic":
            _, v_inputs, vlabels, v_protected = vdata
        else:
            v_inputs, _, vlabels, v_protected = vdata

        voutputs = saved_model(v_inputs)

        # save the predictions and ground truths from each batch for processing
        predictions.append(voutputs.numpy())
        y_labels.append(vlabels.numpy())
        protected.append(v_protected)

# get the performance and fairness metrics
accuracy, precision, recall, fscore = calculate_performance_measures(y_labels, predictions)
eq_odds, eq_oppor, eq_acc = calculate_fairness_measures(y_labels, predictions, protected)

# print out all the metrics
print(f"Model: {SAVED_MODEL_WEIGHTS}\n----------")
print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-score: {fscore}\n----------")
print(f"Equal odds: {eq_odds}\nEqual opportunity: {eq_oppor}\nEqual accuracy: {eq_oppor}\n----------")
