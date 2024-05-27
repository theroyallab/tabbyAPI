from packaging import version
from importlib.metadata import version as package_version
from loguru import logger


def check_exllama_version():
    """Verifies the exllama version"""

    required_version = version.parse("0.1.1")
    current_version = version.parse(package_version("exllamav2").split("+")[0])

    if current_version < required_version:
        raise SystemExit(
            f"ERROR: TabbyAPI requires ExLlamaV2 {required_version} "
            f"or greater. Your current version is {current_version}.\n"
            "Please upgrade your environment by running a start script "
            "(start.bat or start.sh)\n\n"
            "Or you can manually run a requirements update "
            "using the following command:\n\n"
            "For CUDA 12.1:\n"
            "pip install --upgrade .[cu121]\n\n"
            "For CUDA 11.8:\n"
            "pip install --upgrade .[cu118]\n\n"
            "For ROCm:\n"
            "pip install --upgrade .[amd]\n\n"
        )
    else:
        logger.info(f"ExllamaV2 version: {current_version}")
