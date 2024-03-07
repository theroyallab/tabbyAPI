""" Downloader types """
from pydantic import BaseModel
from typing import Optional


class HFDownloadRequest(BaseModel):
    """Represents a HuggingFace download request."""

    repo_id: str
    revision: Optional[str] = "main"
    repo_type: Optional[str] = "model"
    hf_token: Optional[str] = None