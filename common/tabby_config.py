import yaml
import pathlib
from loguru import logger
from typing import Optional

from common.utils import unwrap, merge_dicts


class TabbyConfig:
    network: dict = {}
    logging: dict = {}
    model: dict = {}
    draft_model: dict = {}
    lora: dict = {}
    sampling: dict = {}
    developer: dict = {}
    embeddings: dict = {}

    def __init__(self, arguments: Optional[dict] = None):
        """load the global application config"""

        # config is applied in order of items in the list
        configs = [
            self._from_file(pathlib.Path("config.yml")),
            self._from_args(unwrap(arguments, {})),
        ]

        merged_config = merge_dicts(*configs)

        self.network = unwrap(merged_config.get("network"), {})
        self.logging = unwrap(merged_config.get("logging"), {})
        self.model = unwrap(merged_config.get("model"), {})
        self.draft_model = unwrap(merged_config.get("draft"), {})
        self.lora = unwrap(merged_config.get("draft"), {})
        self.sampling = unwrap(merged_config.get("sampling"), {})
        self.developer = unwrap(merged_config.get("developer"), {})
        self.embeddings = unwrap(merged_config.get("embeddings"), {})

    def _from_file(self, config_path: pathlib.Path):
        """loads config from a given file path"""

        # try loading from file
        try:
            with open(str(config_path.resolve()), "r", encoding="utf8") as config_file:
                return unwrap(yaml.safe_load(config_file), {})
        except FileNotFoundError:
            logger.info("The config.yml file cannot be found")
        except Exception as exc:
            logger.error(
                "The YAML config couldn't load because of "
                f"the following error:\n\n{exc}"
            )

        # if no config file was loaded
        return {}

    def _from_args(self, args: dict):
        """loads config from the provided arguments"""
        config = {}

        config_override = unwrap(args.get("options", {}).get("config"))
        if config_override:
            logger.info("Config file override detected in args.")
            config = self.from_file(pathlib.Path(config_override))
            return config  # Return early if loading from file

        for key in ["network", "model", "logging", "developer", "embeddings"]:
            override = args.get(key)
            if override:
                if key == "logging":
                    # Strip the "log_" prefix from logging keys if present
                    override = {k.replace("log_", ""): v for k, v in override.items()}
                config[key] = override

        return config

    def _from_environment(self):
        """loads configuration from environment variables"""

        # TODO: load config from environment variables
        # this means that we can have host default to 0.0.0.0 in docker for example
        # this would also mean that docker containers no longer require a non
        # default config file to be used
        pass


# Create an empty instance of the shared var to make sure nothing breaks
config: TabbyConfig = TabbyConfig()


def load_config(arguments: dict):
    """Load a populated config class on startup."""

    global shared_config

    shared_config = TabbyConfig(arguments)
