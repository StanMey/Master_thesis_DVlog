import os
import torch

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

from utils.dataloaders import MultimodalEmbeddingsDataset, SyncedMultimodalEmbeddingsDataset
from utils.metrics import calculate_performance_measures, calculate_gender_performance_measures, calculate_fairness_measures
from utils.util import ConfigDict
from utils.util import validate_config, process_config, set_seed


def evaluate_cli(
    config_path: Path = typer.Argument(..., help="Path to config file", exists=True, allow_dash=True),
    model_path: Annotated[Path, typer.Option(help="Path to saved models folder", exists=True, allow_dash=True)] = Path(r"./trained_models"),
    use_gpu: Annotated[int, typer.Option("--gpu-id", "-g", help="GPU ID or -1 for CPU")] = -1,
    unpriv_feature: Annotated[str, typer.Option("--unpriv-feature", "-u", help="The unprivileged fairness feature (m or f)")] = "m",
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="When 'True' prints the measures to console, otherwise returns them as a dict")] = True,
    seed: Annotated[int, typer.Option("--seed", "-s", help="The seed used when randomness is introduced")] = 42
):
    """The CLI function handling the interaction part when the user runs the evaluate.py script.
    """
    evaluate(config_path=config_path, model_path=model_path, use_gpu=use_gpu, unpriv_feature=unpriv_feature, verbose=verbose, seed=seed)


def evaluate(
    config_path: Union[str, Path],
    model_path: Union[str, Path],
    use_gpu: int,
    unpriv_feature: str,
    verbose: bool,
    seed: int
):
    """Function to extract and process the configuration file and setup the models and dataloaders to evaluate the model.
    """
    # check whether the paths exists
    assert os.path.exists(config_path), "Config file does not exist"
    assert os.path.isdir(model_path), "Model path is not a directory"

    # check the unprivileged feature
    assert unpriv_feature in ["m", "f"], "Unprivileged feature should be either 'm' or 'f'"

    # check the config file on completeness
    config = Config().from_disk(config_path)
    validate_config(config)
    config_dict = process_config(config)
    if verbose:
        print(config)

    # begin setting up the model for the evaluation cycle
    model_dir_path = Path(os.path.join(model_path, config_dict.model_name))
    saved_model_path = Path(os.path.join(model_dir_path, f"model_{config_dict.model_name}.pth"))
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

    #TODO setup the device

    # load in the dataset
    if config_dict.encoder1_use_sync:
        # use the synced data loader
        test_data = SyncedMultimodalEmbeddingsDataset("test", config_dict, to_tensor=True, with_protected=True)
    else:
        test_data = MultimodalEmbeddingsDataset("test", config_dict, to_tensor=True, with_protected=True)

    # setup the dataloader
    test_dataloader = DataLoader(test_data, batch_size=config_dict.batch_size, shuffle=True)

    # evaluate the model
    return evaluate_model(saved_model, test_dataloader, config_dict, unpriv_feature, verbose)


def evaluate_model(model, test_dataloader: DataLoader, config_dict: ConfigDict, unpriv_feature: str, verbose: bool) -> Union[None, dict]:
    """Run the actual evaluation process using the model and dataloader.
    """
    predictions = []
    y_labels = []
    protected = []

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for _, vdata in enumerate(test_dataloader):
            
            # get the inputs
            if config_dict.n_modalities == 1:
                # only one modality so unpack the set
                v_inputs, vlabels, v_protected = vdata
            else:
                # more modalities so keep the tuple set
                v_inputs, vlabels, v_protected = vdata[:-2], vdata[-2], vdata[-1]

            # forward
            voutputs = model(v_inputs)

            # save the predictions and ground truths from each batch for processing
            predictions.append(voutputs.numpy())
            y_labels.append(vlabels.numpy())
            protected.append(v_protected)

    # get the performance and fairness metrics
    print(f"Model: {config_dict.model_name}\n----------")

    # calculate the performance and fairness measures
    accuracy, w_precision, w_recall, w_fscore, m_fscore = calculate_performance_measures(y_labels, predictions)
    eq_oppor, eq_acc, fairl_eq_odds, unpriv_stats, priv_stats = calculate_fairness_measures(y_labels, predictions, protected, unprivileged=unpriv_feature)
    gender_metrics = calculate_gender_performance_measures(y_labels, predictions, protected)

    if verbose:
        # print all the calculated measures
        print(f"(weighted) -->\nAccuracy: {accuracy}\nPrecision: {w_precision}\nRecall: {w_recall}\nF1-score: {w_fscore}")
        print(f"(macro) -->\nF1-score: {m_fscore}\n----------")
        print(f"Equal odds (fairlearn): {fairl_eq_odds}\nEqual opportunity: {eq_oppor}\nEqual accuracy: {eq_oppor}\n----------")

        # print the gender-based metrics
        print("Gender-based metrics:\n----------")
        for gender_metric in gender_metrics:
            print("Metrics for label {0}:\n---\nPrecision: {1}\nRecall: {2}\nF1-score: {3}\n----------".format(*gender_metric))
    
    else:
        # return all measures as a dict
        return {
            "model_name": config_dict.model_name,
            "weighted": {
                "accuracy": accuracy,
                "precision": w_precision,
                "recall": w_recall,
                "fscore": w_fscore,
                f"{gender_metrics[0][0]}_fscore": gender_metrics[0][3],
                f"{gender_metrics[1][0]}_fscore": gender_metrics[1][3],
            },
            "macro": {
                "fscore": m_fscore
            },
            "fairness": {
                "fairl_eq_odds": fairl_eq_odds,
                "eq_oppor": eq_oppor,
                "eq_acc": eq_acc,
                "unpriv": {
                    "TPR": unpriv_stats[0],
                    "FPR": unpriv_stats[1]
                },
                "priv": {
                    "TPR": priv_stats[0],
                    "FPR": priv_stats[1]
                }
            }
        }


if __name__ == "__main__":
    typer.run(evaluate_cli)
