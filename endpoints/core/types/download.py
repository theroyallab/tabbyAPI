from pydantic import BaseModel, Field


def _generate_include_list():
    return ["*"]


class DownloadRequest(BaseModel):
    """Parameters for a HuggingFace repo download."""

    repo_id: str
    repo_type: str = "model"
    folder_name: str | None = None
    revision: str | None = None
    token: str | None = None
    include: list[str] = Field(default_factory=_generate_include_list)
    exclude: list[str] = Field(default_factory=list)
    chunk_limit: int | None = None
    timeout: int | None = None


class DownloadResponse(BaseModel):
    """Response for a download request."""

    download_path: str
