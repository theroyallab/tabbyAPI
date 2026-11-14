import asyncio
import json
import pathlib
import shutil
import sys
from huggingface_hub import HfApi
from fnmatch import fnmatch
from common.logger import xlogger
from typing import List, Optional

from common.status_display import status_display
from common.tabby_config import config
from common.utils import unwrap


def _get_repo_info(repo_id, revision, token):
    """Fetches information about a HuggingFace repository."""

    return HfApi().list_repo_files(repo_id, revision=revision or None, token=token or None)


def _get_download_folder(repo_id: str, repo_type: str, folder_name: Optional[str]):
    """Gets the download folder for the repo."""

    if repo_type == "lora":
        download_path = pathlib.Path(config.lora.lora_dir)
    else:
        download_path = pathlib.Path(config.model.model_dir)

    download_path = download_path / (folder_name or repo_id.split("/")[-1])
    return download_path


def _check_exclusions(filename: str, include_patterns: List[str], exclude_patterns: List[str]):
    include_result = any(fnmatch(filename, pattern) for pattern in include_patterns)
    exclude_result = any(fnmatch(filename, pattern) for pattern in exclude_patterns)

    return include_result and not exclude_result


async def hf_repo_download(
    repo_id: str,
    folder_name: Optional[str],
    revision: Optional[str],
    token: Optional[str],
    include: Optional[List[str]],
    exclude: Optional[List[str]],
    chunk_limit: Optional[float] = None,
    timeout: Optional[int] = None,
    repo_type: Optional[str] = "model",
):
    """Gets a repo's information from HuggingFace and downloads it locally."""

    file_list = await asyncio.to_thread(_get_repo_info, repo_id, revision, token)

    if not repo_type:
        if any(
            filename.endswith(("adapter_config.json", "adapter_model.bin"))
            for filename in file_list
        ):
            repo_type = "lora"

    if include or exclude:
        include_patterns = unwrap(include, ["*"])
        exclude_patterns = unwrap(exclude, [])
        file_list = [
            filename
            for filename in file_list
            if _check_exclusions(filename, include_patterns, exclude_patterns)
        ]

    if not file_list:
        raise ValueError(f"File list for repo {repo_id} is empty. Check your filters?")

    download_path = _get_download_folder(repo_id, repo_type, folder_name)

    if download_path.exists():
        raise FileExistsError(
            f"The path {download_path} already exists. Remove the folder and try again."
        )

    download_path.parent.mkdir(parents=True, exist_ok=True)

    xlogger.info(f"Saving {repo_id} to {str(download_path)}")

    try:
        async with status_display.suspended():
            xlogger.info(f"Starting download for {repo_id}")
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                "import json, sys; from huggingface_hub import snapshot_download; "
                "snapshot_download(**json.load(sys.stdin))",
                stdin=asyncio.subprocess.PIPE,
            )
            try:
                await asyncio.wait_for(
                    process.communicate(
                        json.dumps(
                            {
                                "repo_id": repo_id,
                                "revision": revision or None,
                                "token": token or None,
                                "local_dir": str(download_path.resolve()),
                                "allow_patterns": include or None,
                                "ignore_patterns": exclude or None,
                            }
                        ).encode()
                    ),
                    timeout=timeout or None,
                )
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()

            if process.returncode:
                raise RuntimeError(f"Hugging Face download exited with status {process.returncode}")

            xlogger.info(f"Finished download for {repo_id}")
            return download_path
    except (asyncio.CancelledError, Exception):
        if download_path.is_dir():
            shutil.rmtree(download_path)
        raise
