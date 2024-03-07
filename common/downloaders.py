from common.logger import init_logger

logger = init_logger(__name__)
try:
    from huggingface_hub import snapshot_download
except ImportError:
    logger.error(
        "huggingface_hub not installed, HF downloader endpoint will not be functional."
    )
    snapshot_download = None


def hf_download(repo_id, revision, path, api_token):
    if not snapshot_download:
        logger.error("HF downloader is not available.")
        return
    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=path,
        local_dir_use_symlinks=False,
        token=api_token,
    )
