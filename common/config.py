import yaml
import pathlib
from loguru import logger
from mergedeep import merge, Strategy
from typing import Any

from common.utils import unwrap

# Global config dictionary constant
GLOBAL_CONFIG: dict = {}


def load(arguments: dict[str, Any]):
    """load the global application config"""
    global GLOBAL_CONFIG

    # config is applied in order of items in the list
    configs = [
        from_file(pathlib.Path("config.yml")),
        from_environment(),
        from_args(arguments),
    ]

    GLOBAL_CONFIG = merge({}, *configs, strategy=Strategy.REPLACE)

def from_file(config_path: pathlib.Path) -> dict[str, Any]:
    """loads config from a given file path"""

    # try loading from file
    try:
        with open(str(config_path.resolve()), "r", encoding="utf8") as config_file:
            return unwrap(yaml.safe_load(config_file), {})
    except FileNotFoundError:
        logger.info("The config.yml file cannot be found")
    except Exception as exc:
        logger.error(
            f"The YAML config couldn't load because of the following error:\n\n{exc}"
        )

    # if no config file was loaded
    return {}


def from_args(args: dict[str, Any]) -> dict[str, Any]:
    """loads config from the provided arguments"""
    config = {}

    config_override = unwrap(args.get("options", {}).get("config"))
    if config_override:
        logger.info("Config file override detected in args.")
        config = from_file(pathlib.Path(config_override))
        return config  # Return early if loading from file

    for key in ["network", "model", "logging", "developer", "embeddings"]:
        override = args.get(key)
        if override:
            if key == "logging":
                # Strip the "log_" prefix from logging keys if present
                override = {k.replace("log_", ""): v for k, v in override.items()}
            config[key] = override

    return config


def from_environment() -> dict[str, Any]:
    """loads configuration from environment variables"""

    # TODO: load config from environment variables
    # this means that we can have host default to 0.0.0.0 in docker for example
    # this would also mean that docker containers no longer require a non
    # default config file to be used
    return {}


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


def logging_config():
    """Returns the logging config from the global config"""
    return unwrap(GLOBAL_CONFIG.get("logging"), {})


def developer_config():
    """Returns the developer specific config from the global config"""
    return unwrap(GLOBAL_CONFIG.get("developer"), {})


def embeddings_config():
    """Returns the embeddings config from the global config"""
    return unwrap(GLOBAL_CONFIG.get("embeddings"), {})
