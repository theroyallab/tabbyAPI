import yaml
import pathlib

from logger import init_logger
from utils import unwrap

logger = init_logger(__name__)

GLOBAL_CONFIG: dict = {}


def read_config_from_file(config_path: pathlib.Path):
    """Sets the global config from a given file path"""
    global GLOBAL_CONFIG

    try:
        with open(str(config_path), "r", encoding="utf8") as config_file:
            GLOBAL_CONFIG = unwrap(yaml.safe_load(config_file), {})
    except Exception as exc:
        logger.error(
            "The YAML config couldn't load because of the following error: "
            f"\n\n{exc}"
            "\n\nTabbyAPI will start anyway and not parse this config file."
        )
        GLOBAL_CONFIG = {}


def get_model_config():
    """Returns the model config from the global config"""
    return unwrap(GLOBAL_CONFIG.get("model"), {})


def get_draft_model_config():
    """Returns the draft model config from the global config"""
    model_config = unwrap(GLOBAL_CONFIG.get("model"), {})
    return unwrap(model_config.get("draft"), {})


def get_lora_config():
    """Returns the lora config from the global config"""
    model_config = unwrap(GLOBAL_CONFIG.get("model"), {})
    return unwrap(model_config.get("lora"), {})


def get_network_config():
    """Returns the network config from the global config"""
    return unwrap(GLOBAL_CONFIG.get("network"), {})


def get_gen_logging_config():
    """Returns the generation logging config from the global config"""
    return unwrap(GLOBAL_CONFIG.get("logging"), {})
