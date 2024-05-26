import aiofiles
import aiohttp
import asyncio
import math
import pathlib
import shutil
from huggingface_hub import HfApi, hf_hub_url
from fnmatch import fnmatch
from loguru import logger
from rich.progress import Progress
from typing import List, Optional

from common.config import lora_config, model_config
from common.logger import get_progress_bar
from common.utils import unwrap


async def _download_file(
    session: aiohttp.ClientSession,
    repo_item: dict,
    token: Optional[str],
    download_path: pathlib.Path,
    chunk_limit: int,
    progress: Progress,
):
    """Downloads a repo from HuggingFace."""

    filename = repo_item.get("filename")
    url = repo_item.get("url")

    # Default is 2MB
    chunk_limit_bytes = math.ceil(unwrap(chunk_limit, 2000000) * 100000)

    filepath = download_path / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    req_headers = {"Authorization": f"Bearer {token}"} if token else {}

    async with session.get(url, headers=req_headers) as response:
        # TODO: Change to raise errors
        assert response.status == 200

        file_size = int(response.headers["Content-Length"])

        download_task = progress.add_task(
            f"[cyan]Downloading {filename}", total=file_size
        )

        # Chunk limit is 2 MB
        async with aiofiles.open(str(filepath), "wb") as f:
            async for chunk in response.content.iter_chunked(chunk_limit_bytes):
                await f.write(chunk)
                progress.update(download_task, advance=len(chunk))


# Huggingface does not know how async works
def _get_repo_info(repo_id, revision, token):
    """Fetches information about a HuggingFace repository."""

    # None-ish casting of revision and token values
    revision = revision or None
    token = token or None

    api_client = HfApi()
    repo_tree = api_client.list_repo_files(repo_id, revision=revision, token=token)
    return [
        {
            "filename": filename,
            "url": hf_hub_url(repo_id, filename, revision=revision),
        }
        for filename in repo_tree
    ]


def _get_download_folder(repo_id: str, repo_type: str, folder_name: Optional[str]):
    """Gets the download folder for the repo."""

    if repo_type == "lora":
        download_path = pathlib.Path(lora_config().get("lora_dir") or "loras")
    else:
        download_path = pathlib.Path(model_config().get("model_dir") or "models")

    download_path = download_path / (folder_name or repo_id.split("/")[-1])
    return download_path


def _check_exclusions(
    filename: str, include_patterns: List[str], exclude_patterns: List[str]
):
    include_result = any(fnmatch(filename, pattern) for pattern in include_patterns)
    exclude_result = any(fnmatch(filename, pattern) for pattern in exclude_patterns)

    return include_result and not exclude_result


async def hf_repo_download(
    repo_id: str,
    folder_name: Optional[str],
    revision: Optional[str],
    token: Optional[str],
    chunk_limit: Optional[float],
    include: Optional[List[str]],
    exclude: Optional[List[str]],
    repo_type: Optional[str] = "model",
):
    """Gets a repo's information from HuggingFace and downloads it locally."""

    file_list = await asyncio.to_thread(_get_repo_info, repo_id, revision, token)

    # Auto-detect repo type if it isn't provided
    if not repo_type:
        lora_filter = filter(
            lambda repo_item: repo_item.get("filename", "").endswith(
                ("adapter_config.json", "adapter_model.bin")
            )
        )

        if lora_filter:
            repo_type = "lora"

    if include or exclude:
        include_patterns = unwrap(include, ["*"])
        exclude_patterns = unwrap(exclude, [])

        file_list = [
            file
            for file in file_list
            if _check_exclusions(
                file.get("filename"), include_patterns, exclude_patterns
            )
        ]

    if not file_list:
        raise ValueError(f"File list for repo {repo_id} is empty. Check your filters?")

    download_path = _get_download_folder(repo_id, repo_type, folder_name)

    if download_path.exists():
        raise FileExistsError(
            f"The path {download_path} already exists. Remove the folder and try again."
        )

    download_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving {repo_id} to {str(download_path)}")

    try:
        async with aiohttp.ClientSession() as session:
            tasks = []
            logger.info(f"Starting download for {repo_id}")

            progress = get_progress_bar()
            progress.start()

            for repo_item in file_list:
                tasks.append(
                    _download_file(
                        session,
                        repo_item,
                        token=token,
                        download_path=download_path.resolve(),
                        chunk_limit=chunk_limit,
                        progress=progress,
                    )
                )

            await asyncio.gather(*tasks)
            progress.stop()
            logger.info(f"Finished download for {repo_id}")

            return download_path
    except (asyncio.CancelledError, Exception) as exc:
        # Cleanup on cancel
        if download_path.is_dir():
            shutil.rmtree(download_path)
        else:
            download_path.unlink()

        # Stop the progress bar
        progress.stop()

        # Re-raise exception if the task isn't cancelled
        if not isinstance(exc, asyncio.CancelledError):
            raise exc
