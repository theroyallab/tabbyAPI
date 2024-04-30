from pydantic import BaseModel, Field
from typing import List, Optional


def _generate_include_list():
    return ["*"]


class DownloadRequest(BaseModel):
    """Parameters for a HuggingFace repo download."""

    repo_id: str
    repo_type: str = "model"
    folder_name: Optional[str] = None
    revision: Optional[str] = None
    token: Optional[str] = None
    include: List[str] = Field(default_factory=_generate_include_list)
    exclude: List[str] = Field(default_factory=list)
    chunk_limit: Optional[int] = None


class DownloadResponse(BaseModel):
    """Response for a download request."""

    download_path: str
