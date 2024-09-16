import yaml
import pathlib
from loguru import logger
from typing import Optional
from os import getenv

from common.utils import unwrap, merge_dicts
from common.config_models import TabbyConfigModel, generate_config_file


class TabbyConfig(TabbyConfigModel):
    # Persistent defaults
    # TODO: make this pydantic?
    model_defaults: dict = {}

    def load(self, arguments: Optional[dict] = None):
        """Synchronously loads the global application config"""

        # config is applied in order of items in the list
        configs = [
            self._from_file(pathlib.Path("config.yml")),
            self._from_environment(),
            self._from_args(unwrap(arguments, {})),
        ]

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
