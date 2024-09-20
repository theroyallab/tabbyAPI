import platform
from packaging import version
from importlib.metadata import version as package_version
from loguru import logger
from common.optional_dependencies import dependencies


def check_exllama_version():
    """Verifies the exllama version"""

    install_message = (
        "Please update your environment by running an update script "
        "(update_scripts/"
        f"update_deps.{'bat' if platform.system() == 'Windows' else 'sh'})\n\n"
        "Or you can manually run a requirements update "
        "using the following command:\n\n"
        "For CUDA 12.1:\n"
        "pip install --upgrade .[cu121]\n\n"
        "For CUDA 11.8:\n"
        "pip install --upgrade .[cu118]\n\n"
        "For ROCm:\n"
        "pip install --upgrade .[amd]\n\n"
    )

    if not dependencies.exl2:
        raise SystemExit(("Exllamav2 is not installed.\n" + install_message))

    required_version = version.parse("0.2.2")
    current_version = version.parse(package_version("exllamav2").split("+")[0])

    unsupported_message = (
        f"ERROR: TabbyAPI requires ExLlamaV2 {required_version} "
        f"or greater. Your current version is {current_version}.\n" + install_message
    )

    if current_version < required_version:
        raise SystemExit(unsupported_message)
    else:
        logger.info(f"ExllamaV2 version: {current_version}")
