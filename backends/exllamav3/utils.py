import platform
from loguru import logger

def exllama_supports_nccl():
    if platform.system() == "Windows":
        unsupported_message = (
            "The NCCL tensor parallel backend is not supported on Windows."
        )
        logger.warning(unsupported_message)
        return False

    import torch
    return torch.cuda.is_available() and torch.distributed.is_nccl_available()
