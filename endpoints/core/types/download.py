from pydantic import BaseModel, Field
from typing import List, Literal, Optional


def _generate_include_list():
    return ["*"]


class DownloadRequest(BaseModel):
    """Parameters for a HuggingFace repo download."""

    repo_id: str = Field(
        description="The repo ID to download from",
        examples=[
            "royallab/TinyLlama-1.1B-2T-exl2",
            "royallab/LLaMA2-13B-TiefighterLR-exl2",
            "turboderp/Llama-3.1-8B-Instruct-exl2",
        ],
    )
    repo_type: Literal["model", "lora"] = Field("model", description="The model type")
    folder_name: Optional[str] = Field(
        default=None,
        description="The folder name to save the repo to "
        + "(this is used to load the model)",
    )
    revision: Optional[str] = Field(
        default=None, description="The revision to download from"
    )
    token: Optional[str] = Field(
        default=None,
        description="The HuggingFace API token to use, "
        + "required for private/gated repos",
    )
    include: List[str] = Field(
        default_factory=_generate_include_list,
        description="A list of file patterns to include in the download",
    )
    exclude: List[str] = Field(
        default_factory=list,
        description="A list of file patterns to exclude from the download",
    )
    chunk_limit: Optional[int] = Field(
        None, description="The maximum chunk size to download in bytes"
    )
    timeout: Optional[int] = Field(
        None, description="The timeout for the download in seconds"
    )


class DownloadResponse(BaseModel):
    """Response for a download request."""

    download_path: str = Field(description="The path to the downloaded repo")
