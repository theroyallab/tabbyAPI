import json
from loguru import logger

from common.tabby_config import config, generate_config_file
from endpoints.server import export_openapi
from endpoints.utils import do_export_openapi


def branch_to_actions() -> bool:
    """Checks if a optional action needs to be run."""

    if config.actions.export_openapi or do_export_openapi:
        openapi_json = export_openapi()

        with open(config.actions.openapi_export_path, "w") as f:
            f.write(json.dumps(openapi_json))
            logger.info(
                "Successfully wrote OpenAPI spec to "
                + f"{config.actions.openapi_export_path}"
            )
    elif config.actions.export_config:
        generate_config_file(filename=config.actions.config_export_path)
    else:
        # did not branch
        return False

    # branched and ran an action
    return True
