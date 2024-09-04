import yaml
import pathlib
from loguru import logger
from typing import Any

from common.utils import unwrap, merge_dicts

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

    GLOBAL_CONFIG = merge_dicts(*configs)


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


# refactor the get_config functions
def get_config(config: dict[str, any], topic: str) -> callable:
    return lambda: unwrap(config.get(topic), {})


# each of these is a function
model_config = get_config(GLOBAL_CONFIG, "model")
sampling_config = get_config(GLOBAL_CONFIG, "sampling")
draft_model_config = get_config(model_config(), "draft")
lora_config = get_config(model_config(), "lora")
network_config = get_config(GLOBAL_CONFIG, "network")
logging_config = get_config(GLOBAL_CONFIG, "logging")
developer_config = get_config(GLOBAL_CONFIG, "developer")
embeddings_config = get_config(GLOBAL_CONFIG, "embeddings")
