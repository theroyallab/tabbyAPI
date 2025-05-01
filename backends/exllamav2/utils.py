from loguru import logger


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
