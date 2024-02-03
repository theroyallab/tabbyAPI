import yaml
import pathlib

from common.logger import init_logger
from common.utils import unwrap

logger = init_logger(__name__)

GLOBAL_CONFIG: dict = {}


def read_config_from_file(config_path: pathlib.Path):
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


def override_config_from_args(args: dict):
    """Overrides the config based on a dict representation of args"""

    config_override = unwrap(args.get("options", {}).get("config"))
    if config_override:
        logger.info("Attempting to override config.yml from args.")
        read_config_from_file(pathlib.Path(config_override))
        return

    # Network config
    network_override = args.get("network")
    if network_override:
        network_config = get_network_config()
        GLOBAL_CONFIG["network"] = {**network_config, **network_override}

    # Model config
    model_override = args.get("model")
    if model_override:
        model_config = get_model_config()
        GLOBAL_CONFIG["model"] = {**model_config, **model_override}

    # Logging config
    logging_override = args.get("logging")
    if logging_override:
        logging_config = get_gen_logging_config()
        GLOBAL_CONFIG["logging"] = {
            **logging_config,
            **{k.replace("log_", ""): logging_override[k] for k in logging_override},
        }

    developer_override = args.get("developer")
    if developer_override:
        developer_config = get_developer_config()
        GLOBAL_CONFIG["developer"] = {**developer_config, **developer_override}


def get_sampling_config():
    """Returns the sampling parameter config from the global config"""
    return unwrap(GLOBAL_CONFIG.get("sampling"), {})


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


def get_developer_config():
    """Returns the developer specific config from the global config"""
    return unwrap(GLOBAL_CONFIG.get("developer"), {})
