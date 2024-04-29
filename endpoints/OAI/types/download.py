from pydantic import BaseModel
from typing import Optional


class DownloadRequest(BaseModel):
    """Parameters for a HuggingFace repo download."""

    repo_id: str
    repo_type: Optional[str] = "model"
    folder_name: Optional[str] = None
    revision: Optional[str] = None
    token: Optional[str] = None
    chunk_limit: Optional[int] = None


class DownloadResponse(BaseModel):
    """Response for a download request."""

    download_path: str
