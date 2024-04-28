import os
import pandas as pd
import torch
torch.manual_seed(42)

import typer
from typing_extensions import Annotated

# PyTorch TensorBoard support
# from torch.utils.tensorboard import SummaryWriter
from confection import Config
from typing import Union

from pathlib import Path

from evaluate import evaluate
from train import train
from utils.util import validate_config, process_config


def run_cli(
    config_dir: Path = typer.Argument(..., help="Path to directory of config files", exists=True, allow_dash=True),
    output_dir: Annotated[Path, typer.Option(help="Path to saved model folder", exists=True, allow_dash=True)] = Path(r"./trained_models"),
    use_gpu: Annotated[int, typer.Option("--gpu-id", "-g", help="GPU ID or -1 for CPU")] = -1,
    evaluate: Annotated[bool, typer.Option("--evaluate/--train", help="whether to evaluate the models or to train the not yet trained models")] = True,
    unpriv_feature: Annotated[str, typer.Option("--unpriv-feature", "-u", help="The unprivileged fairness feature (m or f) (needed for the evaluation step)")] = "m",
):
    """The CLI function handling the interaction part.
    """
    # check whether the paths exists
    assert os.path.exists(config_dir), "Config file dir does not exist"
    assert os.path.isdir(output_dir), "Output path is not a directory"

    if evaluate:
        # evaluate the models
        print("Begin evaluating")
        evaluate_models(config_dir=config_dir, output_dir=output_dir, use_gpu=use_gpu, unpriv_feature=unpriv_feature)
    else:
        # train the not yet trained models
        print("Begin training")
        train_models(config_dir=config_dir, output_dir=output_dir, use_gpu=use_gpu)


def evaluate_models(
    config_dir: Union[str, Path],
    output_dir: Union[str, Path],
    use_gpu: int,
    unpriv_feature: str
):
    # setup the metric list
    metrics = []

    # get all the config directories
    config_dirs = os.listdir(Path(config_dir))
    trained_models_names = os.listdir(Path(output_dir))
    
    # for each subdirectory go over the configs, extract the names and run the evaluation
    for subdir in config_dirs:

        config_files = os.listdir(os.path.join(config_dir, subdir))
        for config_file in config_files:
            config_path = os.path.join(config_dir, subdir, config_file)
            
            # extract the name of the model and check whether the model exists
            config = Config().from_disk(config_path)
            validate_config(config)
            config_dict = process_config(config)
            model_name = config_dict.model_name

            # check if the model exists
            if model_name in trained_models_names:
                # model exists, so run the evaluation and save the metrics
                metrics.append(evaluate(config_path, output_dir, use_gpu, unpriv_feature=unpriv_feature, verbose=False))

    # extract the metrics
    end_metrics = []
    for metric in metrics:
        weighted = metric.get("weighted")
        fairness = metric.get("fairness")
        end_metrics.append(
            (metric.get("model_name"), weighted.get("precision"), weighted.get("recall"), weighted.get("fscore"), metric.get("macro").get("fscore"),
             weighted.get("m_fscore"), weighted.get("f_fscore"), fairness.get("eq_acc"), fairness.get("eq_odds"), fairness.get("eq_oppor"))
        )

    # build the pandas dataframe, so we can store the results
    df = pd.DataFrame(end_metrics, columns =['Name', 'precision', 'recall', "f1 (weighted)", "f1 (macro)", "f1_m", "f1_f", "Eq accuracy", "Eq odds", "eq opportunity"])
    df.to_csv(os.path.join(output_dir, "metrics.csv"), sep=";")


def train_models(
    config_dir: Union[str, Path],
    output_dir: Union[str, Path],
    use_gpu: int
):
    """Function to extract and process the configuration file and setup the models and dataloaders.
    """
    # get all the config directories
    config_dirs = os.listdir(Path(config_dir))
    trained_models_names = os.listdir(Path(output_dir))

    # for each subdirectory go over the configs, extract the names and check whether the model already exists
    for subdir in config_dirs:

        config_files = os.listdir(os.path.join(config_dir, subdir))
        for config_file in config_files:
            config_path = os.path.join(config_dir, subdir, config_file)
            
            # extract the name of the model and check whether the model exists
            config = Config().from_disk(config_path)
            validate_config(config)
            config_dict = process_config(config)
            model_name = config_dict.model_name
            
            # check if the model exists
            if model_name in trained_models_names:
                print(f"Model {model_name} already exists")
            
            else:
                # we still have to train the model
                print(f"Training model {model_name}")
                train(config_path=config_path, output_path=output_dir, use_gpu=use_gpu)


if __name__ == "__main__":
    typer.run(run_cli)
