import yaml
import pathlib
from inspect import getdoc
from pydantic_core import PydanticUndefined
from loguru import logger
from textwrap import dedent
from typing import Optional
from os import getenv

from common.utils import unwrap, merge_dicts
from common.config_models import BaseConfigModel, TabbyConfigModel


class TabbyConfig(TabbyConfigModel):
    # Persistent defaults
    # TODO: make this pydantic?
    model_defaults: dict = {}

    def load(self, arguments: Optional[dict] = None):
        """Synchronously loads the global application config"""

        # config is applied in order of items in the list
        arguments_dict = unwrap(arguments, {})
        configs = [self._from_environment(), self._from_args(arguments_dict)]

        # If actions aren't present, also look from the file
        # TODO: Change logic if file loading requires actions in the future
        if not arguments_dict.get("actions"):
            configs.insert(0, self._from_file(pathlib.Path("config.yml")))

        merged_config = merge_dicts(*configs)

        # validate and update config
        merged_config_model = TabbyConfigModel.model_validate(merged_config)
        for field in TabbyConfigModel.model_fields.keys():
            value = getattr(merged_config_model, field)
            setattr(self, field, value)

        # Set model defaults dict once to prevent on-demand reconstruction
        # TODO: clean this up a bit
        for field in self.model.use_as_default:
            if hasattr(self.model, field):
                self.model_defaults[field] = getattr(config.model, field)
            elif hasattr(self.draft_model, field):
                self.model_defaults[field] = getattr(config.draft_model, field)
            else:
                logger.error(
                    f"invalid item {field} in config option `model.use_as_default`"
                )

    def _from_file(self, config_path: pathlib.Path):
        """loads config from a given file path"""

        legacy = False
        cfg = {}

        # try loading from file
        try:
            with open(str(config_path.resolve()), "r", encoding="utf8") as config_file:
                cfg = yaml.safe_load(config_file)

                # NOTE: Remove migration wrapper after a period of time
                # load legacy config files

                # Model config migration
                model_cfg = unwrap(cfg.get("model"), {})

                if model_cfg.get("draft"):
                    legacy = True
                    cfg["draft"] = model_cfg["draft"]

                if model_cfg.get("lora"):
                    legacy = True
                    cfg["lora"] = model_cfg["lora"]

                # Logging config migration
                # This will catch the majority of legacy config files
                logging_cfg = unwrap(cfg.get("logging"), {})
                unmigrated_log_keys = [
                    key for key in logging_cfg.keys() if not key.startswith("log_")
                ]
                if unmigrated_log_keys:
                    legacy = True
                    for key in unmigrated_log_keys:
                        cfg["logging"][f"log_{key}"] = cfg["logging"][key]
                        del cfg["logging"][key]
        except FileNotFoundError:
            logger.info(f"The '{config_path.name}' file cannot be found")
        except Exception as exc:
            logger.error(
                f"The YAML config from '{config_path.name}' couldn't load because of "
                f"the following error:\n\n{exc}"
            )

        if legacy:
            logger.warning(
                "Legacy config.yml file detected. Attempting auto-migration."
            )

            # Create a temporary base config model
            new_cfg = TabbyConfigModel.model_validate(cfg)

            try:
                config_path.rename(f"{config_path}.bak")
                generate_config_file(model=new_cfg, filename=config_path)
                logger.info(
                    "Auto-migration successful. "
                    'The old configuration is stored in "config.yml.bak".'
                )
            except Exception as e:
                logger.error(
                    f"Auto-migration failed because of: {e}\n\n"
                    "Reverted all changes.\n"
                    "Either fix your config.yml and restart or\n"
                    "Delete your old YAML file and create a new "
                    'config by copying "config_sample.yml" to "config.yml".'
                )

                # Restore the old config
                config_path.unlink(missing_ok=True)
                pathlib.Path(f"{config_path}.bak").rename(config_path)

                # Don't use the partially loaded config
                logger.warning("Starting with no config loaded.")
                return {}

        return unwrap(cfg, {})

    def _from_args(self, args: dict):
        """loads config from the provided arguments"""
        config = {}

        config_override = unwrap(args.get("options", {}).get("config"))
        if config_override:
            logger.info("Config file override detected in args.")
            config = self._from_file(pathlib.Path(config_override))
            return config  # Return early if loading from file

        for key in TabbyConfigModel.model_fields.keys():
            override = args.get(key)
            if override:
                config[key] = override

        return config

    def _from_environment(self):
        """loads configuration from environment variables"""

        config = {}

        for field_name in TabbyConfigModel.model_fields.keys():
            section_config = {}
            for sub_field_name in getattr(
                TabbyConfigModel(), field_name
            ).model_fields.keys():
                setting = getenv(f"TABBY_{field_name}_{sub_field_name}".upper(), None)
                if setting is not None:
                    section_config[sub_field_name] = setting

            config[field_name] = section_config

        return config


# Create an empty instance of the config class
config: TabbyConfig = TabbyConfig()


# TODO: Possibly switch to ruamel.yaml for a more native implementation
def generate_config_file(
    model: BaseConfigModel = None,
    filename: str = "config_sample.yml",
    indentation: int = 2,
) -> None:
    """Creates a config.yml file from Pydantic models."""

    # Add a cleaned up preamble
    preamble = """
    # Sample YAML file for configuration.
    # Comment and uncomment values as needed.
    # Every value has a default within the application.
    # This file serves to be a drop in for config.yml

    # Unless specified in the comments, DO NOT put these options in quotes!
    # You can use https://www.yamllint.com/ if you want to check your YAML formatting.\n
    """

    # Trim and cleanup preamble
    yaml = dedent(preamble).lstrip()

    schema = unwrap(model, TabbyConfigModel())

    # TODO: Make the disordered iteration look cleaner
    iter_once = False
    for field, field_data in schema.model_fields.items():
        # Fetch from the existing model class if it's passed
        # Probably can use this on schema too, but play it safe
        if model and hasattr(model, field):
            subfield_model = getattr(model, field)
        else:
            subfield_model = field_data.default_factory()

        if not subfield_model._metadata.include_in_config:
            continue

        # Since the list is out of order with the length
        # Add newlines from the beginning once one iteration finishes
        # This is a sanity check for formatting
        if iter_once:
            yaml += "\n"
        else:
            iter_once = True

        for line in getdoc(subfield_model).splitlines():
            yaml += f"# {line}\n"

        yaml += f"{field}:\n"

        sub_iter_once = False
        for subfield, subfield_data in subfield_model.model_fields.items():
            # Same logic as iter_once
            if sub_iter_once:
                yaml += "\n"
            else:
                sub_iter_once = True

            # If a value already exists, use it
            if hasattr(subfield_model, subfield):
                value = getattr(subfield_model, subfield)
            elif subfield_data.default_factory:
                value = subfield_data.default_factory()
            else:
                value = subfield_data.default

            value = value if value is not None else ""
            value = value if value is not PydanticUndefined else ""

            for line in subfield_data.description.splitlines():
                yaml += f"{' ' * indentation}# {line}\n"

            yaml += f"{' ' * indentation}{subfield}: {value}\n"

    with open(filename, "w") as f:
        f.write(yaml)
