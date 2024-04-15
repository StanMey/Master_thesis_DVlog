import torch

# PyTorch TensorBoard support
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from pathlib import Path

from utils.dataloaders import BaseDVlogDataset, UnimodalEmbeddingsDataset
from utils.metrics import calculate_performance_measures, calculate_gender_performance_measures, calculate_fairness_measures
from models.model import BimodalDVlogModel, UnimodalDVlogModel


# HARDCODE SOME VARIABLES
# batch size: 32
# epochs: 50
# learning rate: 0.0002
# sequence length (t): 596
BATCH_SIZE = 32
SEQUENCE_LENGTH = 104
DIM_MODEL = 256
N_HEADS_UNIMODAL = 8
N_HEADS_CROSS = 16
USE_GPU = True
USE_STD = False
SAVED_MODEL_WEIGHTS = "unimodal_w2v_seq-avg_t104"
SAVED_MODEL_PATH = Path(f"trained_models/model_{SAVED_MODEL_WEIGHTS}")

# evaluation parameters
modality = "uni" # can choose between acoustic, visual, uni, or both
dataset = "test" # can choose between 'test', 'train', or 'val'
fairness_unprivileged = "m"
feature_name = "w2v_seq_avg"
visual_feature_dim = 136
acoustic_feature_dim = 25
uni_feature_dim = 300

# do the checks over the parameters
assert modality in ["visual", "acoustic", "uni", "both"], f"Modality type not in choices: {modality}"
assert dataset in ["test", "train", "val"], f"Chosen dataset not in choices: {dataset}"
assert SAVED_MODEL_PATH.is_file(), f"Saved model not found: {SAVED_MODEL_PATH}"


# setup the paths
annotations_file = Path(r"./dataset/dvlog_labels_v2.csv")
data_dir = Path(r"./dataset/dvlog-dataset")
# data_dir = Path(r"./dataset/pdem-dataset")
data_dir = Path(r"./dataset/embeddings-dataset")


# load the saved version of the model
if modality == "acoustic":
    # acoustic unimodel
    saved_model = UnimodalDVlogModel(data_shape=(SEQUENCE_LENGTH, acoustic_feature_dim), d_model=DIM_MODEL, n_heads=N_HEADS_UNIMODAL, use_std=USE_STD)
elif modality == "visual":
    # visual unimodel
    saved_model = UnimodalDVlogModel(data_shape=(SEQUENCE_LENGTH, visual_feature_dim), d_model=DIM_MODEL, n_heads=N_HEADS_UNIMODAL, use_std=USE_STD)
elif modality == "uni":
    # unimodal model
    saved_model = UnimodalDVlogModel(data_shape=(SEQUENCE_LENGTH, uni_feature_dim), d_model=DIM_MODEL, n_heads=N_HEADS_UNIMODAL, use_std=USE_STD)
else:
    # bimodal model
    saved_model = BimodalDVlogModel(d_model=DIM_MODEL, uni_n_heads=N_HEADS_UNIMODAL, cross_n_heads=N_HEADS_CROSS, use_std=USE_STD)

saved_model.load_state_dict(torch.load(SAVED_MODEL_PATH))
# Set the model to evaluation mode
saved_model.eval()

# setup the device
# device = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")

# load in the dataset
if modality == "uni":
    eval_dataset = UnimodalEmbeddingsDataset(annotations_file, data_dir, dataset, feature_name=feature_name, sequence_length=SEQUENCE_LENGTH, to_tensor=True, with_protected=True)
else:
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
        elif modality == "visual":
            v_inputs, _, vlabels, v_protected = vdata
        elif modality == "uni":
            v_inputs, vlabels, v_protected = vdata
        else:
            v_inputs_v, v_inputs_a, vlabels, v_protected = vdata

        # forward + backward + optimize
        if modality == "both":
            voutputs = saved_model((v_inputs_a, v_inputs_v))
        else:
            voutputs = saved_model(v_inputs)

        # save the predictions and ground truths from each batch for processing
        predictions.append(voutputs.numpy())
        y_labels.append(vlabels.numpy())
        protected.append(v_protected)

# get the performance and fairness metrics
accuracy, w_precision, w_recall, w_fscore, m_fscore = calculate_performance_measures(y_labels, predictions)
eq_odds, eq_oppor, eq_acc = calculate_fairness_measures(y_labels, predictions, protected, fairness_unprivileged)

# print out all the metrics
print(f"Model: {SAVED_MODEL_WEIGHTS}\n----------")
print(f"(weighted) -->\nAccuracy: {accuracy}\nPrecision: {w_precision}\nRecall: {w_recall}\nF1-score: {w_fscore}")
print(f"(macro) -->\nF1-score: {m_fscore}\n----------")
print(f"Equal odds: {eq_odds}\nEqual opportunity: {eq_oppor}\nEqual accuracy: {eq_oppor}\n----------")

# print out the gender-based metrics
print("Gender-based metrics:\n----------")
gender_metrics = calculate_gender_performance_measures(y_labels, predictions, protected)
for gender_metric in gender_metrics:
    print("Metrics for label {0}:\n---\nPrecision: {1}\nRecall: {2}\nF1-score: {3}\n----------".format(*gender_metric))
