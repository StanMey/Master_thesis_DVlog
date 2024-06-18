import os
import torch
import numpy as np

import typer
from typing_extensions import Annotated

# PyTorch TensorBoard support
# from torch.utils.tensorboard import SummaryWriter
from confection import Config
from typing import Union

from torch.utils.data import DataLoader
from pathlib import Path
from itertools import chain

from models.model import UnimodalDVlogModel, BimodalDVlogModel
from models.trimodal_model import TrimodalDVlogModel

from utils.dataloaders import MultimodalEmbeddingsDataset, SyncedMultimodalEmbeddingsDataset
from utils.metrics import calculate_performance_measures, calculate_gender_performance_measures, calculate_fairness_measures
from utils.util import ConfigDict
from utils.util import validate_config, process_config, set_seed


def evaluate_cli(
    config_path: Path = typer.Argument(..., help="Path to config file", exists=True, allow_dash=True),
    models_path: Annotated[Path, typer.Option(help="Path to saved models folder", exists=True, allow_dash=True)] = Path(r"./trained_models"),
    unpriv_feature: Annotated[str, typer.Option("--unpriv-feature", "-u", help="The unprivileged fairness feature (m or f)")] = "m",
    dataset: Annotated[str, typer.Option("--dataset", help="")] = "test",
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="When 'True' prints the measures to console, otherwise returns them as a dict")] = True,
    seed: Annotated[int, typer.Option("--seed", "-s", help="The seed used when randomness is introduced")] = 42,
    gender_spec: Annotated[str, typer.Option("--gspec", help="Whether we want to run the model on only one gender ['m' or 'f']")] = None
):
    """The CLI function handling the interaction part when the user runs the evaluate.py script.
    """
    evaluate(config_path=config_path, models_path=models_path, unpriv_feature=unpriv_feature, verbose=verbose, seed=seed, gender_spec=gender_spec)


def evaluate(
    config_path: Union[str, Path],
    models_path: Union[str, Path],
    unpriv_feature: str,
    dataset: str,
    verbose: bool,
    get_raw_predictions: bool = False,
    seed: int = 42,
    gender_spec: str = None,

):
    """Function to extract and process the configuration file and setup the models and dataloaders to evaluate the model.
    """
    # check whether the paths exists
    assert os.path.exists(config_path), "Config file does not exist"
    assert os.path.isdir(models_path), "Model path is not a directory"

    # check the unprivileged feature
    assert unpriv_feature in ["m", "f"], "Unprivileged feature should be either 'm' or 'f'"
    assert gender_spec in ["m", "f", None], "Gender specific token not correct"
    assert dataset in ["train", "test", "val"], "Selected dataset not correct"

    # check the config file on completeness
    config = Config().from_disk(config_path)
    validate_config(config)
    config_dict = process_config(config)
    if verbose:
        print(config)

    # begin setting up the model for the evaluation cycle
    model_dir_path = Path(os.path.join(models_path, config_dict.model_name))
    saved_model_path = Path(os.path.join(model_dir_path, f"model_{config_dict.model_name}_seed{seed}.pth"))
    assert model_dir_path.is_dir(), f"Saved model directory not found: {model_dir_path}"
    assert saved_model_path.is_file(), f"Saved model not found: {saved_model_path}"

    # set the seed
    set_seed(seed)

    # setup the initial model
    if config_dict.n_modalities == 1:
        # initiate an unimodal model
        saved_model = UnimodalDVlogModel((config_dict.sequence_length, config_dict.encoder1_dim),
                                   d_model=config_dict.dim_model, n_heads=config_dict.uni_n_heads, use_std=config_dict.detectlayer_use_std)

    elif config_dict.n_modalities == 2:
        # initiate a bimodal model
        saved_model = BimodalDVlogModel((config_dict.sequence_length, config_dict.encoder1_dim), (config_dict.sequence_length, config_dict.encoder2_dim), cross_type=config_dict.multi_bi_type,
                                   d_model=config_dict.dim_model, uni_n_heads=config_dict.uni_n_heads, cross_n_heads=config_dict.multi_n_heads, use_std=config_dict.detectlayer_use_std)

    else:
        # initiate a trimodal model
        seq_length = config_dict.sequence_length
        saved_model = TrimodalDVlogModel((seq_length, config_dict.encoder1_dim), (seq_length, config_dict.encoder2_dim), (seq_length, config_dict.encoder3_dim), cross_type=config_dict.multi_tri_type,
                                   d_model=config_dict.dim_model, uni_n_heads=config_dict.uni_n_heads, cross_n_heads=config_dict.multi_n_heads, use_std=config_dict.detectlayer_use_std)
    
    # load in the parameters and set the model to evaluation mode
    saved_model.load_state_dict(torch.load(saved_model_path))
    saved_model.eval()

    # load in the dataset
    if config_dict.encoder1_use_sync:
        # use the synced data loader
        test_data = SyncedMultimodalEmbeddingsDataset(dataset, config_dict, to_tensor=True, with_protected=True)
    else:
        test_data = MultimodalEmbeddingsDataset(dataset, config_dict, to_tensor=True, with_protected=True, gender_spec=gender_spec)

    # setup the dataloader
    test_dataloader = DataLoader(test_data, batch_size=config_dict.batch_size, shuffle=True)

    # evaluate the model
    return evaluate_model(saved_model, test_dataloader, config_dict, unpriv_feature, verbose, seed, get_raw_preds=get_raw_predictions, gender_spec=gender_spec)


def evaluate_model(model, test_dataloader: DataLoader, config_dict: ConfigDict, unpriv_feature: str, verbose: bool, seed: int, get_raw_preds: bool, gender_spec: str = None) -> Union[None, dict]:
    """Run the actual evaluation process using the model and dataloader.
    """
    predictions = []
    y_labels = []
    protected = []
    video_ids = []

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for _, vdata in enumerate(test_dataloader):
            
            # get the inputs
            if config_dict.n_modalities == 1:
                # only one modality so unpack the set
                v_inputs, vlabels, v_protected, v_video_id = vdata
            else:
                # more modalities so keep the tuple set
                v_inputs, vlabels, v_protected, v_video_id = vdata[:-3], vdata[-3], vdata[-2], vdata[-1]

            # forward
            voutputs = model(v_inputs)

            # save the predictions and ground truths from each batch for processing
            predictions.append(voutputs.numpy())
            y_labels.append(vlabels.numpy())
            protected.append(v_protected)
            video_ids.append(v_video_id)

    # get the performance and fairness metrics
    print(f"Model: {config_dict.model_name} with seed: {seed}\n----------")

    # put all the predictions and ground truths to an numpy array (flatten them all)
    predictions = np.concatenate(predictions)
    y_labels = np.concatenate(y_labels)
    protected = np.array(list(chain.from_iterable(protected)))
    video_ids = np.array(list(chain.from_iterable(video_ids)))

    # calculate the performance and fairness measures
    accuracy, w_precision, w_recall, w_fscore, _ = calculate_performance_measures(y_labels, predictions)

    if not gender_spec:
        # if we evaluate only for one gender, doing fairness measures and gender performance does not make sense
        # retrieve all fairness details
        eq_oppor, eq_acc, pred_equal, unpriv_stats, priv_stats = calculate_fairness_measures(y_labels, predictions, protected, unprivileged=unpriv_feature)

        gender_metrics = calculate_gender_performance_measures(y_labels, predictions, protected)

    if verbose:
        # print all the calculated measures
        print(f"(weighted) -->\nAccuracy: {accuracy}\nPrecision: {w_precision}\nRecall: {w_recall}\nF1-score: {w_fscore}")

        if not gender_spec:
            print(f"Equal opportunity: {eq_oppor}\nPredictive equality: {pred_equal}\nEqual accuracy: {eq_oppor}\n----------")

            # print the gender-based metrics
            print("Gender-based metrics:\n----------")
            for gender_metric in gender_metrics:
                print("Metrics for label {0}:\n---\nPrecision: {1}\nRecall: {2}\nF1-score: {3}\n----------".format(*gender_metric))
    
    elif get_raw_preds:
        # return the raw predictions (each value is a 2d array because of the dataloader)
        return predictions, y_labels, protected, video_ids
    
    else:
        # return all measures as a dict
        measure_dict = {
                "model_name": config_dict.model_name,
                "model_seed": seed,
                "weighted": {
                    "accuracy": accuracy,
                    "precision": w_precision,
                    "recall": w_recall,
                    "fscore": w_fscore,
                }}

        if not gender_spec:
            # add all extra to the measurements_dict
            measure_dict["weighted"][f"{gender_metrics[0][0]}_fscore"] = gender_metrics[0][3]
            measure_dict["weighted"][f"{gender_metrics[1][0]}_fscore"] = gender_metrics[1][3]
            
            # save all fairness-based measures
            measure_dict["fairness"] = {
                "eq_oppor": eq_oppor,
                "pred_equal": pred_equal,
                "eq_acc": eq_acc,
                "unpriv": {
                    "TPR": unpriv_stats[0],
                    "FPR": unpriv_stats[1]
                },
                "priv": {
                    "TPR": priv_stats[0],
                    "FPR": priv_stats[1]
                }}

        return measure_dict


if __name__ == "__main__":
    typer.run(evaluate_cli)
