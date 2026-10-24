"""Server-side implementations of the built-in web tools.

Both tools fetch external content, so every input is treated as hostile:
URLs are validated and resolved before any request, redirects are never
followed, responses are size-capped, and only text is extracted. The
SearXNG instance and its token come from tools_config.yml, never from
the request.
"""

import asyncio
import ipaddress
import json
import socket
from html.parser import HTMLParser
from typing import List, Optional, Tuple
from urllib.parse import quote_plus, urlparse

import aiohttp
from fastapi import HTTPException

from common import tools_config
from common.logger import xlogger

# Hard ceilings by value: they are constants and never change across reloads.
# The working set of options is read through the module (tools_config.TOOLS_CONFIG)
# so that a hot reload, which replaces the object, is seen by in-flight code.
from common.tools_config import (
    HARD_MAX_PAGE_CHARS,
    HARD_MAX_RESULT_CHARS,
    HARD_MAX_RESULTS,
    HARD_TIMEOUT,
)
from endpoints.OAI.types.web_tools import (
    FetchURLResponse,
    SearchResult,
    WebSearchResponse,
)

# Never download more than this, whatever the client asked for
MAX_DOWNLOAD_BYTES = 1_048_576

TEXT_PREFIXES = ("text/", "application/json", "application/xml", "application/xhtml")


def _clamp(value: Optional[int], default: int, hard_limit: int, minimum: int) -> int:
    """Per-request overrides can lower a limit but never raise it."""

    if value is None:
        return default
    return max(minimum, min(value, hard_limit))


def _blocked_ip(ip: str) -> bool:
    """True for addresses a server must not connect to on behalf of a client."""

    addr = ipaddress.ip_address(ip)
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


async def validate_public_url(url: str, allow_private: bool = False) -> str:
    """
    Check a URL the model or a client supplied before fetching it.

    Only http and https are accepted, and every address the hostname
    resolves to must be public unless allow_private is set. The check runs
    before the request; a hostile hostname that rebinds mid-request is a
    residual risk documented in the tool calling docs.
    """

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(400, "Only http and https URLs are allowed.")
    if not parsed.hostname:
        raise HTTPException(400, "The URL has no hostname.")

    try:
        loop = asyncio.get_running_loop()
        infos = await loop.getaddrinfo(parsed.hostname, None, family=socket.AF_UNSPEC)
    except socket.gaierror as exc:
        raise HTTPException(400, f"Could not resolve {parsed.hostname}: {exc}") from exc

    if not allow_private:
        for info in infos:
            if _blocked_ip(info[4][0]):
                raise HTTPException(
                    403, "Private, loopback and link-local addresses are not allowed."
                )
    return url


def _extract_text(html: str) -> Tuple[Optional[str], str]:
    """Turn an HTML page into plain text, dropping script and style content."""

    class Extractor(HTMLParser):
        SKIP = {"script", "style", "head", "noscript", "template", "svg"}

        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.parts: List[str] = []
            self.title = ""
            self._skip = 0
            self._in_title = False

        def handle_starttag(self, tag, attrs):
            if tag in self.SKIP:
                self._skip += 1
            elif tag == "title":
                self._in_title = True

        def handle_endtag(self, tag):
            if tag in self.SKIP and self._skip:
                self._skip -= 1
            elif tag == "title":
                self._in_title = False

        def handle_data(self, data):
            if self._in_title:
                self.title += data
            elif not self._skip and data.strip():
                self.parts.append(data.strip())

    extractor = Extractor()
    try:
        extractor.feed(html)
    except Exception:
        pass  # Malformed HTML still yields whatever was parsed
    return extractor.title.strip() or None, " ".join(extractor.parts)


def _clean_snippet(text: Optional[str], limit: int) -> str:
    """Collapse whitespace, stringify non-text values, and cap a snippet's length."""

    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return " ".join(text.split())[:limit]


async def searxng_search(
    query: str,
    max_results: Optional[int],
    max_result_chars: Optional[int],
    categories: Optional[str] = None,
    time_range: Optional[str] = None,
) -> WebSearchResponse:
    """Run a search against the configured SearXNG instance."""

    searxng = tools_config.TOOLS_CONFIG.searxng
    if not searxng.enabled:
        raise HTTPException(503, "web_search is disabled in tools_config.yml.")
    if not searxng.url:
        raise HTTPException(503, "searxng.url is not set in tools_config.yml.")

    limit = _clamp(max_results, searxng.max_results, HARD_MAX_RESULTS, 1)
    chars = _clamp(max_result_chars, searxng.max_result_chars, HARD_MAX_RESULT_CHARS, 200)

    search_url = f"{searxng.url.rstrip('/')}/search?q={quote_plus(query)}&format=json"
    if categories:
        search_url += f"&categories={quote_plus(categories)}"
    if time_range:
        search_url += f"&time_range={quote_plus(time_range)}"
    headers = {"Accept": "application/json"}
    if searxng.token:
        headers["Authorization"] = f"Bearer {searxng.token}"

    timeout = aiohttp.ClientTimeout(total=min(searxng.timeout, HARD_TIMEOUT))
    try:
        async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
            async with session.get(search_url, headers=headers, allow_redirects=False) as response:
                if response.status != 200:
                    xlogger.warning(f"SearXNG search failed: status {response.status}.")
                    raise HTTPException(
                        502,
                        f"SearXNG returned {response.status}. Public instances often "
                        "refuse format=json; enable it in the instance's settings.yml.",
                    )
                # aiohttp's StreamReader.read(n) returns whatever is buffered at
                # call time, so chunked bodies arrive one chunk per call: keep
                # reading until the cap or EOF instead of trusting one call.
                raw = b""
                while len(raw) <= MAX_DOWNLOAD_BYTES:
                    chunk = await response.content.read(MAX_DOWNLOAD_BYTES + 1)
                    if not chunk:
                        break
                    raw += chunk
                if len(raw) > MAX_DOWNLOAD_BYTES:
                    xlogger.warning("SearXNG search failed: response too large.")
                    raise HTTPException(502, "The search response is too large.")
                try:
                    payload = json.loads(raw.decode("utf8", errors="replace"))
                except json.JSONDecodeError as exc:
                    xlogger.warning("SearXNG search failed: invalid JSON.")
                    raise HTTPException(502, "SearXNG returned invalid JSON.") from exc
                if not isinstance(payload, dict):
                    xlogger.warning("SearXNG search failed: unexpected JSON shape.")
                    raise HTTPException(502, "SearXNG returned an unexpected JSON shape.")
    except asyncio.TimeoutError as exc:
        xlogger.warning("SearXNG search timed out.")
        raise HTTPException(502, "The search timed out.") from exc
    except aiohttp.ClientError as exc:
        xlogger.warning(f"SearXNG search failed: {exc}")
        # The instance's address comes from tools_config.yml and stays out of
        # client-facing errors.
        raise HTTPException(502, "Could not reach SearXNG.") from exc

    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raw_results = []
    results = [
        SearchResult(
            title=_clean_snippet(item.get("title"), chars),
            # Hostile search payloads must not stuff the model's context with
            # oversized URL strings; real result URLs stay far below this cap.
            url=str(item.get("url") or "")[:1024],
            snippet=_clean_snippet(item.get("content"), chars),
        )
        for item in raw_results
        if isinstance(item, dict)
    ][:limit]
    return WebSearchResponse(query=query, results=results)


async def fetch_page(url: str, max_page_chars: Optional[int]) -> FetchURLResponse:
    """Download a page by URL and return its text."""

    fetch = tools_config.TOOLS_CONFIG.fetch_url
    if not fetch.enabled:
        raise HTTPException(503, "fetch_url is disabled in tools_config.yml.")

    url = await validate_public_url(url, allow_private=fetch.allow_private_urls)
    chars = _clamp(max_page_chars, fetch.max_page_chars, HARD_MAX_PAGE_CHARS, 500)

    timeout = aiohttp.ClientTimeout(total=min(fetch.timeout, HARD_TIMEOUT))
    try:
        async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
            async with session.get(
                url,
                headers={"Accept": "text/html, text/plain, application/json"},
                allow_redirects=False,
            ) as response:
                if 300 <= response.status < 400:
                    target = response.headers.get("Location", "")
                    xlogger.warning(f"Page fetch failed: redirect (status {response.status}).")
                    raise HTTPException(
                        502,
                        f"The page redirects to {target}. Ask for that URL "
                        "directly if you want its content.",
                    )
                if response.status != 200:
                    xlogger.warning(f"Page fetch failed: status {response.status}.")
                    raise HTTPException(502, f"The server returned {response.status}.")
                ctype = response.headers.get("Content-Type", "")
                if not ctype.startswith(TEXT_PREFIXES):
                    xlogger.warning("Page fetch failed: non-text content type.")
                    raise HTTPException(502, "The URL does not return text content.")
                # Same chunked-body rule as the search path above: read until the
                # cap or EOF, never trust a single read(n) call.
                raw = b""
                while len(raw) <= MAX_DOWNLOAD_BYTES:
                    chunk = await response.content.read(MAX_DOWNLOAD_BYTES + 1)
                    if not chunk:
                        break
                    raw += chunk
                if len(raw) > MAX_DOWNLOAD_BYTES:
                    xlogger.warning("Page fetch failed: page too large.")
                    raise HTTPException(502, "The page is too large to fetch.")
    except asyncio.TimeoutError as exc:
        xlogger.warning("Page fetch timed out.")
        raise HTTPException(502, "The page fetch timed out.") from exc
    except aiohttp.ClientError as exc:
        xlogger.warning(f"Page fetch failed: {exc}")
        raise HTTPException(502, f"Could not fetch the page: {exc}") from exc

    body = raw.decode("utf8", errors="replace")
    if ctype.startswith("application/json"):
        try:
            title, text = None, json.dumps(json.loads(body), indent=2)[:chars]
        except json.JSONDecodeError:
            title, text = None, body[:chars]
    else:
        title, text = _extract_text(body)
        text = text[:chars]
    return FetchURLResponse(url=url, title=title, text=text)
