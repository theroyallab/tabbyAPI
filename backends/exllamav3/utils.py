import platform
import torch
from common.logger import xlogger


def exllama_supports_nccl():
    if platform.system() == "Windows":
        unsupported_message = "The NCCL tensor parallel backend is not supported on Windows."
        xlogger.warning(unsupported_message)
        return False

    return torch.cuda.is_available() and torch.distributed.is_nccl_available()
