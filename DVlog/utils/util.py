
def validate_config(config: dict):
    """_summary_

    :param config: .
    :type config: dict
    """
    # check for the existence of certain types in the config
    for tags in [("general",), ("paths",), ("paths", "annotations_file"), ("training",), ("model",), ("model", "n_modalities"), ("model", "encoder"), ("model", "detection_layer")]:
        assert keys_exists(config, *tags), f"keys {tags} missing in config file"
    
    # check if the amount of encoders coincides with the amount of modalities and if everything has been entered correctly
    for x_encoder in range(1, int(get_config_value(config, "model", "n_modalities") + 1)):
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
