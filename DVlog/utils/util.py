import os
import random
import numpy as np
import torch

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
    assert os.path.exists(annotations_file), f"Annotations file could not be found -> {annotations_file}"

    # check if the amount of encoders coincides with the amount of modalities and if everything has been entered correctly
    for x_encoder in range(1, n_modalities + 1):
        # check if the encoder has been defined
        assert keys_exists(config, "model", "encoder", str(x_encoder)), f"Encoder {x_encoder} not defined properly"

        # check whether the encoder has a feature name and dimension
        for key in ["feature_name", "feature_dim", "data_dir"]:
            assert keys_exists(config, "model", "encoder", str(x_encoder), key), f"Key {key} for encoder {x_encoder} not defined properly"
    
    # if we do do a sync function over the first encoder check if we have a sync file in the config
    if keys_exists(config, "model", "encoder", "1", "apply_sync"):
        assert keys_exists(config, "paths", "sync_file"), f"no 'sync_file' section in the config file"

        # check whether the path to the sync file exists
        sync_file = Path(get_config_value(config, "paths", "sync_file"))
        os.path.exists(sync_file), f"Annotations file could not be found -> {sync_file}"

    # if we do have two modalities, check whether the type of cross-embedding is defined
    if n_modalities == 2:
        assert keys_exists(config, "model", "multimodal", "bi_type"), f"no 'bi_type' defined in the config file"
        chosen_option, available_bioptions = get_config_value(config, "model", "multimodal", "bi_type"), ["cross", "concat"]
        assert chosen_option in available_bioptions, f"bi_type {chosen_option} invalid; please choose one of {available_bioptions}"

    # if we do have three modalities, check whether the type of cross-embedding is defined
    if n_modalities == 3:
        assert keys_exists(config, "model", "multimodal", "tri_type"), f"no 'tri_type' defined in the config file"
        chosen_option, available_trioptions = get_config_value(config, "model", "multimodal", "tri_type"), ["crisscross", "layered", "concat"]
        assert chosen_option in available_trioptions, f"tri_type {chosen_option} invalid; please choose one of {available_trioptions}"
    
    # if we do do a some bias mitigation, check whether it is implemented
    if keys_exists(config, "training", "bias_mit"):
        chosen_option, available_biasoptions = get_config_value(config, "training", "bias_mit"), ["mixfeat", "oversample"]
        assert chosen_option in available_biasoptions, f"Bias mitigation option {chosen_option} invalid; please choose one of {available_biasoptions}"
        assert 1 <= n_modalities <= 2, f"Bias mitigation approaches only implemented 1 or 2 modalities, not for {n_modalities}"

        # if we do choose mixfeat check if the corresponding mixfeat type is implemented
        if chosen_option == "mixfeat":
            mixfeat_option, available_mixf_options = get_config_value(config, "training", "mixfeat_type"),  ['group_upsample', 'mixgender_upsample', 'subgroup_upsample', 'synthetic', 'synthetic_mixgendered']
            assert mixfeat_option in available_mixf_options, f"Bias mitigation option {mixfeat_option} invalid; please choose one of {available_mixf_options}"




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
        self.general_data_dir = Path(get_config_value(self.config, "paths", "data_dir"))
        self.sync_file = None if not keys_exists(self.config, "paths", "sync_file") else Path(get_config_value(self.config, "paths", "sync_file"))

        # retrieve the model specific variables
        self.n_modalities = int(get_config_value(self.config, "model", "n_modalities"))
        self.sequence_length = 596 if not keys_exists(self.config, "model", "sequence_length") else get_config_value(self.config, "model", "sequence_length")
        self.dim_model = 256 if not keys_exists(self.config, "model", "dim_model") else get_config_value(self.config, "model", "dim_model")
        self.uni_n_heads = 8 if not keys_exists(self.config, "model", "encoder", "n_heads") else get_config_value(self.config, "model", "encoder", "n_heads")
        self.multi_n_heads = 16 if not keys_exists(self.config, "model", "multimodal", "n_heads") else get_config_value(self.config, "model", "multimodal", "n_heads")
        self.detectlayer_use_std = False if not keys_exists(self.config, "model", "detection_layer", "use_std") else get_config_value(self.config, "model", "detection_layer", "use_std")

        self.multi_bi_type = None if not keys_exists(self.config, "model", "multimodal", "bi_type") else get_config_value(self.config, "model", "multimodal", "bi_type")
        self.multi_tri_type = None if not keys_exists(self.config, "model", "multimodal", "tri_type") else get_config_value(self.config, "model", "multimodal", "tri_type")
        self.bias_mit = None if not keys_exists(self.config, "training", "bias_mit") else get_config_value(self.config, "training", "bias_mit")
        self.mixfeat_type = None if not keys_exists(self.config, "training", "mixfeat_type") else get_config_value(self.config, "training", "mixfeat_type")

        # set up the training variables (here we do have to check each feature since we otherwise can use default values)
        self.batch_size = 32 if not keys_exists(self.config, "training", "batch_size") else get_config_value(self.config, "training", "batch_size")
        self.epochs = 50 if not keys_exists(self.config, "training", "epochs") else get_config_value(self.config, "training", "epochs")
        self.learning_rate = 0.0002 if not keys_exists(self.config, "training", "learning_rate") else get_config_value(self.config, "training", "learning_rate")

        # setup the data directories (begin with the first one and then check if we have others)
        self.encoder1 = get_config_value(self.config, "model", "encoder", "1")
        self.encoder1_feature_name = self.encoder1.get("feature_name")
        self.encoder1_dim = self.encoder1.get("feature_dim")
        self.encoder1_data_dir = Path(self.encoder1.get("data_dir"))
        self.encoder1_use_sync = False if not keys_exists(self.config, "model", "encoder", "1", "apply_sync") else get_config_value(self.config, "model", "encoder", "1", "apply_sync")

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
    

    def to_dict(self) -> dict:
        config_dict = self.config

        # overwrite all values which might have been overwritten with a default value
        config_dict["model"]['dim_model'] = self.dim_model
        config_dict["model"]['sequence_length'] = self.sequence_length
        config_dict["model"]["encoder"]["n_heads"] = self.uni_n_heads
        config_dict["model"]["multimodal"]["n_heads"] = self.multi_n_heads
        config_dict["model"]["detection_layer"]["use_std"] = self.detectlayer_use_std

        config_dict["training"]['epochs'] = self.epochs
        config_dict["training"]['batch_size'] = self.batch_size
        config_dict["training"]['learning_rate'] = self.learning_rate

        return config_dict

    def __repr__(self) -> str:
        return str(self.to_dict())
        


def process_config(config: dict) -> ConfigDict:
    """Small function which builds the config dict (implemented for sanity reasons)
    """
    return ConfigDict(config)


def set_seed(seed: int = 42) -> None:
    """Sets the seed for all the involved libraries in a PyTorch pipeline.
    https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy?galleryTag=pytorch

    :param seed: The value of the seed, defaults to 42
    :type seed: int, optional
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


