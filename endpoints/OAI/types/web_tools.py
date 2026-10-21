"""Request and response types for the built-in web tools."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class WebSearchRequest(BaseModel):
    """Parameters for a SearXNG search."""

    query: str = Field(..., min_length=1, max_length=2000)
    max_results: Optional[int] = Field(
        None, ge=1, description="Clamped to the server's hard limit."
    )
    max_result_chars: Optional[int] = Field(
        None, ge=200, description="Clamped to the server's hard limit."
    )
    categories: Optional[str] = Field(
        None,
        max_length=64,
        pattern=r"^[A-Za-z][A-Za-z0-9_ ,+-]*$",
        description=(
            "SearXNG category to search in, e.g. general, news, images, "
            "science, social media. Instance-dependent."
        ),
    )
    time_range: Optional[Literal["day", "week", "month", "year"]] = Field(
        None, description="Restrict results to the last day, week, month or year."
    )


class SearchResult(BaseModel):
    """One search result, text only."""

    title: str = ""
    url: str = ""
    snippet: str = ""


class WebSearchResponse(BaseModel):
    """Search results for the model."""

    query: str
    results: List[SearchResult] = Field(default_factory=list)


class FetchURLRequest(BaseModel):
    """Parameters for a page fetch."""

    url: str = Field(..., min_length=1, max_length=2000)
    max_page_chars: Optional[int] = Field(
        None, ge=500, description="Clamped to the server's hard limit."
    )


class FetchURLResponse(BaseModel):
    """Extracted page text for the model."""

    url: str
    title: Optional[str] = None
    text: str = ""
