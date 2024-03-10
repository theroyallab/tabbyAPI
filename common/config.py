import yaml
import pathlib
from loguru import logger

from common.utils import unwrap

# Global config dictionary constant
GLOBAL_CONFIG: dict = {}


def from_file(config_path: pathlib.Path):
    """Sets the global config from a given file path"""
    global GLOBAL_CONFIG

    try:
        with open(str(config_path.resolve()), "r", encoding="utf8") as config_file:
            GLOBAL_CONFIG = unwrap(yaml.safe_load(config_file), {})
    except Exception as exc:
        logger.error(
            "The YAML config couldn't load because of the following error: "
            f"\n\n{exc}"
            "\n\nTabbyAPI will start anyway and not parse this config file."
        )
        GLOBAL_CONFIG = {}


def from_args(args: dict):
    """Overrides the config based on a dict representation of args"""

    config_override = unwrap(args.get("options", {}).get("config"))
    if config_override:
        logger.info("Attempting to override config.yml from args.")
        from_file(pathlib.Path(config_override))
        return

    # Network config
    network_override = args.get("network")
    if network_override:
        cur_network_config = network_config()
        GLOBAL_CONFIG["network"] = {**cur_network_config, **network_override}

    # Model config
    model_override = args.get("model")
    if model_override:
        cur_model_config = model_config()
        GLOBAL_CONFIG["model"] = {**cur_model_config, **model_override}

    # Generation Logging config
    gen_logging_override = args.get("logging")
    if gen_logging_override:
        cur_gen_logging_config = gen_logging_config()
        GLOBAL_CONFIG["logging"] = {
            **cur_gen_logging_config,
            **{
                k.replace("log_", ""): gen_logging_override[k]
                for k in gen_logging_override
            },
        }

    developer_override = args.get("developer")
    if developer_override:
        cur_developer_config = developer_config()
        GLOBAL_CONFIG["developer"] = {**cur_developer_config, **developer_override}


def sampling_config():
    """Returns the sampling parameter config from the global config"""
    return unwrap(GLOBAL_CONFIG.get("sampling"), {})


def model_config():
    """Returns the model config from the global config"""
    return unwrap(GLOBAL_CONFIG.get("model"), {})


def draft_model_config():
    """Returns the draft model config from the global config"""
    model_config = unwrap(GLOBAL_CONFIG.get("model"), {})
    return unwrap(model_config.get("draft"), {})


def lora_config():
    """Returns the lora config from the global config"""
    model_config = unwrap(GLOBAL_CONFIG.get("model"), {})
    return unwrap(model_config.get("lora"), {})


def network_config():
    """Returns the network config from the global config"""
    return unwrap(GLOBAL_CONFIG.get("network"), {})


def gen_logging_config():
    """Returns the generation logging config from the global config"""
    return unwrap(GLOBAL_CONFIG.get("logging"), {})


def developer_config():
    """Returns the developer specific config from the global config"""
    return unwrap(GLOBAL_CONFIG.get("developer"), {})
