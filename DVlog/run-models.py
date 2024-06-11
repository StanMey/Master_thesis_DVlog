import os
import pandas as pd
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
    models_path: Annotated[Path, typer.Option(help="Path to saved model folder or where to save the trained models", exists=True, allow_dash=True)] = Path(r"./trained_models"),
    use_gpu: Annotated[int, typer.Option("--gpu-id", "-g", help="GPU ID or -1 for CPU")] = -1,
    evaluate: Annotated[bool, typer.Option("--evaluate/--train", help="whether to evaluate the models or to train the not yet trained models")] = True,
    unpriv_feature: Annotated[str, typer.Option("--unpriv-feature", "-u", help="The unprivileged fairness feature (m or f) (needed for the evaluation step)")] = "m",
    seed: Annotated[int, typer.Option("--seed", "-s", help="The seed used when randomness is introduced")] = 42
):
    """The CLI function handling the interaction part.
    """
    # check whether the paths exists
    assert os.path.exists(config_dir), "Config file dir does not exist"
    assert os.path.isdir(models_path), "Output path is not a directory"

    if evaluate:
        # evaluate the models
        print("Begin evaluating")
        evaluate_models(config_dir=config_dir, models_path=models_path, unpriv_feature=unpriv_feature)
    else:
        # train the not yet trained models
        print("Begin training")
        train_models(config_dir=config_dir, output_dir=models_path, use_gpu=use_gpu, seed=seed)


def evaluate_models(
    config_dir: Union[str, Path],
    models_path: Union[str, Path],
    unpriv_feature: str
):
    # setup the metric list
    metrics = []

    # get all the config directories
    config_dirs = os.listdir(Path(config_dir))
    trained_models_names = os.listdir(Path(models_path))
    
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
                # model exists, so get all the trained model's names
                model_variants = [f for f in os.listdir(os.path.join(models_path, model_name)) if f.endswith(".pth")]

                for variant in model_variants:
                    filename, _ = os.path.splitext(variant)
                    train_seed = int(filename.split("_seed")[-1])

                    # run the evaluation and save the metrics
                    metrics.append(evaluate(config_path, models_path, unpriv_feature=unpriv_feature, verbose=False, seed=train_seed))

    # extract the metrics
    end_metrics = []
    for metric in metrics:
        weighted = metric.get("weighted")
        fairness = metric.get("fairness")

        # get all metrics and put them to floats
        float_metrics = (
            weighted.get("precision"), weighted.get("recall"), weighted.get("fscore"), weighted.get("m_fscore"), weighted.get("f_fscore"),
            fairness.get("eq_acc"), fairness.get("eq_oppor"), fairness.get("pred_equal"),
            fairness.get("unpriv").get("TPR"), fairness.get("unpriv").get("FPR"), fairness.get("priv").get("TPR"), fairness.get("priv").get("FPR"))
        float_metrics = tuple(map(float, float_metrics))
        end_metrics.append((metric.get("model_name"), metric.get("model_seed"), *float_metrics))

    # build the pandas dataframe, so we can store the results
    df = pd.DataFrame(end_metrics, columns=['name', 'seed', 'precision', 'recall', "f1", "f1_m", "f1_f", "Eq accuracy", "eq opportunity", "pred equality", "unpriv_TPR", "unpriv_FPR", "priv_TPR", "priv_FPR"])
    df.to_csv(os.path.join(models_path, "metrics.csv"), sep=";")

    # remove the seed value so groupby operation becomes straightforward
    df.drop(["seed"], axis=1, inplace=True)
    grouped_avg = df.groupby("name").mean()
    grouped_avg.to_csv(os.path.join(models_path, "avg_metrics.csv"), sep=";")

    grouped_std = df.groupby("name").std()
    grouped_std.to_csv(os.path.join(models_path, "std_metrics.csv"), sep=";")


def train_models(
    config_dir: Union[str, Path],
    output_dir: Union[str, Path],
    use_gpu: int,
    seed: int
):
    """Function to extract and process the configuration file and setup the models and dataloaders.
    """
    # get all the config directories
    config_dirs = os.listdir(Path(config_dir))

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
            trained_model_name = os.path.join(Path(output_dir), model_name, f"model_{config_dict.model_name}_seed{seed}.pth")
            
            # check if the model exists
            if os.path.isfile(trained_model_name):
                print(f"Model {model_name}_seed{seed} already exists")
            else:
                # we still have to train the model
                print(f"Training model {model_name}_seed{seed}")
                train(config_path=config_path, output_path=output_dir, use_gpu=use_gpu, seed=seed)


if __name__ == "__main__":
    typer.run(run_cli)
