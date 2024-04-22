import os

from pathlib import Path


CONFIG_CHECKLIST = [("general",), ("paths",), ("paths", "annotations_file"), ("training",), ("model",), ("model", "n_modalities"),
                    ("model", "encoder"), ("model", "detection_layer")]


def validate_config(config: dict):
    """_summary_

    :param config: .
    :type config: dict
    """
    # check for the existence of certain types in the config
    for tags in CONFIG_CHECKLIST:
        assert keys_exists(config, *tags), f"keys {tags} missing in config file"
    
    # check if n_modalities is within range (1, 3)
    n_modalities = int(get_config_value(config, "model", "n_modalities"))
    assert 1 <= n_modalities <= 3, f"Model does not support {n_modalities} modalities, only 1, 2 or 3."

    # check whether the annotations_file exists
    annotations_file = Path(get_config_value(config, "paths", "annotations_file"))
    assert os.path.exists(), f"Annotations file could not be found -> {annotations_file}"

    # check if the amount of encoders coincides with the amount of modalities and if everything has been entered correctly
    for x_encoder in range(1, n_modalities + 1):
        # check if the encoder has been defined
        assert keys_exists(config, "model", "encoder", str(x_encoder)), f"Encoder {x_encoder} not defined properly"

        # check whether the encoder has a feature name and dimension
        for key in ["feature_name", "feature_dim"]:
            assert keys_exists(config, "model", "encoder", str(x_encoder), key), f"Key {key} for encoder {x_encoder} not defined properly"


def keys_exists(config: dict, *keys) -> bool:
    """Check if *keys (nested) exists in `config` (dict).
    https://stackoverflow.com/questions/43491287/elegant-way-to-check-if-a-nested-key-exists-in-a-dict

    :param config: .
    :type config: dict
    """
    _element = config
    for key in keys:
        try:
            _element = _element[key]
        except KeyError:
            return False
    return True


def get_config_value(config: dict, *keys):
    """Retrieve *keys (nested) from (nested) `config` (dict).

    :param config: .
    :type config: dict
    """
    _element = config
    for key in keys:
        try:
            _element = _element[key]
        except KeyError:
            raise KeyError(f"Key does not exist: {keys}")
    return _element


class ConfigDict:
    """Class for storing training related stuff.
    Takes in the dict from the configuration file and builds an config object which also handles values which are not given using default values.

    # PAPER VARIABLES (will be used as default values below)
    # batch size: 32
    # epochs: 50
    # learning rate: 0.0002
    # sequence length (t): 596
    # SEED = 42
    """

    def __init__(self, config: dict) -> None:
        self.config = config

        # setup the overall variables
        self.model_name = 1
        self.annotations_file = 1
        self.n_modalities = 1

        # set up the training variables
        self.batch_size = 1
        self.epochs = 1
        self.learning_rate = 1
        self.sequence_length = 1
        self.seed = 1

        # setup the data directories (begin with the first one and then check if we have others)
        self.n = 1

        if self.n_modalities == 2:
            ...
        
        if self.n_modalities == 3:
            ...



def process_config(config: dict) -> ConfigDict:
    return ConfigDict(config)