from packaging import version
from importlib.metadata import version as package_version

from common.logger import init_logger

logger = init_logger(__name__)


def check_exllama_version():
    """Verifies the exllama version"""

    required_version = version.parse("0.0.13.post2")
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
            "pip install --upgrade -r requirements.txt\n\n"
            "For CUDA 11.8:\n"
            "pip install --upgrade -r requirements-cu118.txt\n\n"
            "For ROCm:\n"
            "pip install --upgrade -r requirements-amd.txt\n\n"
        )
    else:
        logger.info(f"ExllamaV2 version: {current_version}")
