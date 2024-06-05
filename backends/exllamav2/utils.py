from packaging import version
from importlib.metadata import PackageNotFoundError, version as package_version
from loguru import logger
import torch


def check_exllama_version():
    """Verifies the exllama version"""

    required_version = version.parse("0.1.4")
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


def hardware_supports_flash_attn(gpu_device_list: list[int]):
    """
    Check whether all GPUs in list support FA2

    Compute capability < 8 is not supported by FA2
    AMD is also unsupported until ROCm updates its FA2 fork
    """

    min_compute_capability = min(
        torch.cuda.get_device_capability(device=device_idx)[0]
        for device_idx in gpu_device_list
    )
    if torch.version.hip or min_compute_capability < 8:
        return False
    else:
        return True


def supports_paged_attn():
    """Check whether the user's flash-attn version supports paged mode"""

    required_version = version.parse("2.5.7")
    try:
        current_version = version.parse(package_version("flash-attn").split("+")[0])
    except PackageNotFoundError:
        return False

    if current_version < required_version:
        return False
    else:
        return True
