import os

from pathlib import Path


CONFIG_CHECKLIST = [("general",), ("general", "model_name"), ("paths",), ("paths", "annotations_file"), ("paths", "data_dir"), ("training",),
                    ("model",), ("model", "n_modalities"), ("model", "encoder"), ("model", "multimodal"), ("model", "detection_layer")]


def validate_config(config: dict):
    """Function to validate the given configuration file according to some pre-determined rules.

    :param config: The converted config file.
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
        for key in ["feature_name", "feature_dim", "data_dir"]:
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

    # OTHER DEFAULTS
    # dimension model: 256
    # n-heads for multimodal: 16
    """

    def __init__(self, config: dict) -> None:
        self.config = config

        # setup the overall variables
        self.model_name = get_config_value(self.config, "general", "model_name")
        self.annotations_file = Path(get_config_value(self.config, "paths", "annotations_file"))

        # retrieve the model specific variables
        self.n_modalities = int(get_config_value(self.config, "model", "n_modalities"))
        self.sequence_length = 596 if not keys_exists(self.config, "model", "sequence_length") else get_config_value(self.config, "model", "sequence_length")
        self.dim_model = 256 if not keys_exists(self.config, "model", "dim_model") else get_config_value(self.config, "model", "dim_model")
        self.uni_n_heads = 16 if not keys_exists(self.config, "model", "encoder", "n_heads") else get_config_value(self.config, "model", "encoder", "n_heads")
        self.multi_n_heads = 16 if not keys_exists(self.config, "model", "multimodal", "n_heads") else get_config_value(self.config, "model", "multimodal", "n_heads")
        self.detectlayer_use_std = False if not keys_exists(self.config, "model", "detection_layer", "use_std") else get_config_value(self.config, "model", "detection_layer", "use_std")

        # set up the training variables (here we do have to check each feature since we otherwise can use default values)
        self.batch_size = 32 if not keys_exists(self.config, "training", "batch_size") else get_config_value(self.config, "training", "batch_size")
        self.epochs = 50 if not keys_exists(self.config, "training", "epochs") else get_config_value(self.config, "training", "epochs")
        self.learning_rate = 0.0002 if not keys_exists(self.config, "training", "learning_rate") else get_config_value(self.config, "training", "learning_rate")
        self.seed = 42 if not keys_exists(self.config, "training", "seed") else get_config_value(self.config, "training", "seed")

        # setup the data directories (begin with the first one and then check if we have others)
        self.encoder1 = get_config_value(self.config, "model", "encoder", "1")
        self.encoder1_feature_name = self.encoder1.get("feature_name")
        self.encoder1_dim = self.encoder1.get("feature_dim")
        self.encoder1_data_dir = Path(self.encoder1.get("data_dir"))

        if self.n_modalities >= 2:
            # specifically check for larger than 2 (we also want the second encoder when we have 3 modalities)
            self.encoder2 = get_config_value(self.config, "model", "encoder", "2")
            self.encoder2_feature_name = self.encoder2.get("feature_name")
            self.encoder2_dim = self.encoder2.get("feature_dim")
            self.encoder2_data_dir = Path(self.encoder2.get("data_dir"))
        
        if self.n_modalities == 3:
            self.encoder3 = get_config_value(self.config, "model", "encoder", "3")
            self.encoder3_feature_name = self.encoder3.get("feature_name")
            self.encoder3_dim = self.encoder3.get("feature_dim")
            self.encoder3_data_dir = Path(self.encoder3.get("data_dir"))


def process_config(config: dict) -> ConfigDict:
    """Small function which builds the config dict (implemented for sanity reasons)
    """
    return ConfigDict(config)