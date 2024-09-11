import platform
import torch
from packaging import version
from importlib.metadata import PackageNotFoundError, version as package_version
from loguru import logger


def check_exllama_version():
    """Verifies the exllama version"""

    required_version = version.parse("0.2.1")
    current_version = version.parse(package_version("exllamav2").split("+")[0])

    unsupported_message = (
        f"ERROR: TabbyAPI requires ExLlamaV2 {required_version} "
        f"or greater. Your current version is {current_version}.\n"
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

    if current_version < required_version:
        raise SystemExit(unsupported_message)
    else:
        logger.info(f"ExllamaV2 version: {current_version}")


def hardware_supports_flash_attn(gpu_device_list: list[int]):
    """
    Check whether all GPUs in list support FA2

    Compute capability < 8 is not supported by FA2
    AMD is also unsupported until ROCm updates its FA2 fork
    """

    # Logged message if unsupported
    unsupported_message = (
        "An unsupported GPU is found in this configuration. "
        "Switching to compatibility mode. \n"
        "This disables parallel batching "
        "and features that rely on it (ex. CFG). \n"
        "To disable compatability mode, all GPUs must be ampere "
        "(30 series) or newer. AMD GPUs are not supported."
    )

    min_compute_capability = min(
        torch.cuda.get_device_capability(device=device_idx)[0]
        for device_idx in gpu_device_list
    )

    if torch.version.hip or min_compute_capability < 8:
        logger.warning(unsupported_message)
        return False
    else:
        return True


def supports_paged_attn():
    """Check whether the user's flash-attn version supports paged mode"""

    # Logged message if unsupported
    unsupported_message = (
        "Flash attention version >=2.5.7 "
        "is required to use paged attention. "
        "Switching to compatibility mode. \n"
        "This disables parallel batching "
        "and features that rely on it (ex. CFG). \n"
        "Please upgrade your environment by running an update script "
        "(update_scripts/"
        f"update_deps.{'bat' if platform.system() == 'Windows' else 'sh'})\n\n"
        "Or you can manually run a requirements update "
        "using the following command:\n\n"
        "For CUDA 12.1:\n"
        "pip install --upgrade .[cu121]\n\n"
        "For CUDA 11.8:\n"
        "pip install --upgrade .[cu118]\n\n"
        "NOTE: Windows users must use CUDA 12.x to use flash-attn."
    )

    required_version = version.parse("2.5.7")
    try:
        current_version = version.parse(package_version("flash-attn").split("+")[0])
    except PackageNotFoundError:
        logger.warning(unsupported_message)
        return False

    if current_version < required_version:
        logger.warning(unsupported_message)
        return False
    else:
        return True


def exllama_disabled_flash_attn(no_flash_attn: bool):
    unsupported_message = (
        "ExllamaV2 has disabled Flash Attention. \n"
        "Please see the above logs for warnings/errors. \n"
        "Switching to compatibility mode. \n"
        "This disables parallel batching "
        "and features that rely on it (ex. CFG). \n"
    )

    if no_flash_attn:
        logger.warning(unsupported_message)

    return no_flash_attn
