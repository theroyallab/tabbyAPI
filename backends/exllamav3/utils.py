import platform
from loguru import logger


def exllama_supports_nccl():
    if platform.system() != "Windows":
        return False

    unsupported_message = (
        "The NCCL tensor parallel backend is not supported on Windows. \n"
        "Switching to native backend."
    )
    logger.warning(unsupported_message)

    return True
