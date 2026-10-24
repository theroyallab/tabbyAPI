"""The built-in web tools: URL guards, result shaping, and endpoint auth."""

import asyncio
import inspect
import json
from types import SimpleNamespace

import aiohttp
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from common import auth, tools_config
from endpoints.OAI.types.web_tools import WebSearchRequest
from endpoints.OAI.utils import web_tools
from endpoints.OAI.utils.web_tools import (
    MAX_DOWNLOAD_BYTES,
    _blocked_ip,
    _clamp,
    _clean_snippet,
    _extract_text,
    validate_public_url,
)
from endpoints.server import setup_app


def _searxng_cfg(**overrides):
    searxng = {
        "enabled": True,
        "url": "http://searx.invalid",
        "token": None,
        "timeout": 10.0,
        "max_results": 8,
        "max_result_chars": 500,
    }
    searxng.update(overrides)
    return SimpleNamespace(searxng=SimpleNamespace(**searxng))


def _fetch_cfg(**overrides):
    fetch_url = {
        "enabled": True,
        "timeout": 10.0,
        "max_page_chars": 8000,
        "allow_private_urls": True,
    }
    fetch_url.update(overrides)
    return SimpleNamespace(fetch_url=SimpleNamespace(**fetch_url))


class _FakeContent:
    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    async def read(self, limit: int) -> bytes:
        chunk = self._data[self._pos : self._pos + limit]
        self._pos += len(chunk)
        return chunk


class _FakeResponse:
    def __init__(self, status=200, data=b"", headers=None):
        self.status = status
        self.headers = headers or {}
        self.content = _FakeContent(data)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _patch_session(monkeypatch, response=None, error=None):
    urls = []

    class _FakeSession:
        def __init__(self, *args, **kwargs):
            assert set(kwargs) == {"timeout", "trust_env"}, (
                f"unexpected session kwargs: {sorted(kwargs)}"
            )
            assert kwargs["trust_env"] is False, "environment proxies must never be used"

        def get(self, url, *, headers=None, allow_redirects=True, **kwargs):
            # Mirror the real aiohttp.ClientSession.get signature so a wrong
            # or misspelled keyword (e.g. httpx's follow_redirects) fails the
            # test instead of being silently swallowed by **kwargs.
            assert allow_redirects is False, "redirects must never be followed"
            assert not kwargs, f"unexpected aiohttp kwargs: {sorted(kwargs)}"
            urls.append(url)
            if error is not None:
                raise error
            return response

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(web_tools.aiohttp, "ClientSession", _FakeSession)
    return urls


def test_aiohttp_redirect_kwarg_name():
    # Guards the _FakeSession.get signature against a future aiohttp rename.
    params = inspect.signature(aiohttp.ClientSession.get).parameters
    assert "allow_redirects" in params
    assert "follow_redirects" not in params


def test_blocked_ips():
    for ip in (
        "127.0.0.1",
        "10.1.2.3",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.1.1",
        "::1",
        "fe80::1",
        "fc00::1",
        "0.0.0.0",
    ):
        assert _blocked_ip(ip)
    for ip in ("8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"):
        assert not _blocked_ip(ip)


def test_clamp_lowers_only():
    assert _clamp(None, 8, 20, 1) == 8
    assert _clamp(5, 8, 20, 1) == 5
    assert _clamp(99, 8, 20, 1) == 20
    assert _clamp(0, 8, 20, 1) == 1


def test_clean_snippet():
    assert _clean_snippet("  a\n  b  ", 10) == "a b"
    assert _clean_snippet("x" * 50, 10) == "x" * 10
    assert _clean_snippet(None, 10) == ""


def test_clean_snippet_coerces_non_strings():
    assert _clean_snippet(5, 10) == "5"
    assert "'a'" in _clean_snippet({"a": 1}, 100)


def test_extract_text_drops_scripts():
    title, text = _extract_text(
        "<html><head><title>T</title><style>x</style></head>"
        "<body><script>bad()</script><p>hello</p></body></html>"
    )
    assert title == "T"
    assert text == "hello"


def test_extract_text_without_title():
    title, text = _extract_text("<html><body><p>hi</p></body></html>")
    assert title is None
    assert text == "hi"


def test_validate_public_url_rejects_private():
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(validate_public_url("http://127.0.0.1:5000/x"))
    assert exc_info.value.status_code == 403
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(validate_public_url("ftp://example.com/x"))
    assert exc_info.value.status_code == 400


def test_search_non_dict_payload_is_502(monkeypatch):
    monkeypatch.setattr(tools_config, "TOOLS_CONFIG", _searxng_cfg())
    _patch_session(monkeypatch, _FakeResponse(data=json.dumps([1, 2]).encode()))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(web_tools.searxng_search("q", None, None))
    assert exc_info.value.status_code == 502


def test_search_skips_non_dict_results(monkeypatch):
    monkeypatch.setattr(tools_config, "TOOLS_CONFIG", _searxng_cfg())
    payload = {"results": ["junk", 5, {"title": "ok", "url": "http://x", "content": "c"}]}
    _patch_session(monkeypatch, _FakeResponse(data=json.dumps(payload).encode()))
    response = asyncio.run(web_tools.searxng_search("q", None, None))
    assert [r.title for r in response.results] == ["ok"]


def test_search_results_string_gives_empty(monkeypatch):
    monkeypatch.setattr(tools_config, "TOOLS_CONFIG", _searxng_cfg())
    _patch_session(monkeypatch, _FakeResponse(data=json.dumps({"results": "nope"}).encode()))
    response = asyncio.run(web_tools.searxng_search("q", None, None))
    assert response.results == []


def test_search_scalar_fields_coerced(monkeypatch):
    monkeypatch.setattr(tools_config, "TOOLS_CONFIG", _searxng_cfg())
    payload = {"results": [{"title": 5, "url": "http://x", "content": None}]}
    _patch_session(monkeypatch, _FakeResponse(data=json.dumps(payload).encode()))
    response = asyncio.run(web_tools.searxng_search("q", None, None))
    assert len(response.results) == 1
    assert response.results[0].title == "5"
    assert response.results[0].snippet == ""


def test_search_reports_timeout_not_unreachable(monkeypatch):
    monkeypatch.setattr(tools_config, "TOOLS_CONFIG", _searxng_cfg())
    _patch_session(monkeypatch, error=aiohttp.ServerTimeoutError())
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(web_tools.searxng_search("q", None, None))
    assert exc_info.value.status_code == 502
    assert "timed out" in exc_info.value.detail


def test_search_omits_categories_and_time_range_by_default(monkeypatch):
    monkeypatch.setattr(tools_config, "TOOLS_CONFIG", _searxng_cfg())
    urls = _patch_session(monkeypatch, _FakeResponse(data=b'{"results": []}'))
    asyncio.run(web_tools.searxng_search("hello world", None, None))
    assert "categories" not in urls[0]
    assert "time_range" not in urls[0]
    assert "q=hello+world" in urls[0]


def test_search_forwards_categories_and_time_range(monkeypatch):
    monkeypatch.setattr(tools_config, "TOOLS_CONFIG", _searxng_cfg())
    urls = _patch_session(monkeypatch, _FakeResponse(data=b'{"results": []}'))
    asyncio.run(web_tools.searxng_search("q", None, None, "news", "day"))
    assert "categories=news" in urls[0]
    assert "time_range=day" in urls[0]


def test_search_time_range_only(monkeypatch):
    monkeypatch.setattr(tools_config, "TOOLS_CONFIG", _searxng_cfg())
    urls = _patch_session(monkeypatch, _FakeResponse(data=b'{"results": []}'))
    asyncio.run(web_tools.searxng_search("q", None, None, None, "week"))
    assert "time_range=week" in urls[0]
    assert "categories" not in urls[0]


def test_search_encodes_categories_with_space(monkeypatch):
    monkeypatch.setattr(tools_config, "TOOLS_CONFIG", _searxng_cfg())
    urls = _patch_session(monkeypatch, _FakeResponse(data=b'{"results": []}'))
    asyncio.run(web_tools.searxng_search("q", None, None, "social media", None))
    assert "categories=social+media" in urls[0]


def test_search_url_field_is_capped(monkeypatch):
    monkeypatch.setattr(tools_config, "TOOLS_CONFIG", _searxng_cfg())
    payload = {"results": [{"title": "t", "url": "http://x/" + "y" * 5000, "content": "c"}]}
    _patch_session(monkeypatch, _FakeResponse(data=json.dumps(payload).encode()))
    response = asyncio.run(web_tools.searxng_search("q", None, None))
    assert len(response.results[0].url) == 1024


def test_search_response_too_large_is_502(monkeypatch):
    monkeypatch.setattr(tools_config, "TOOLS_CONFIG", _searxng_cfg())
    big = b'{"results": []}' + b" " * (MAX_DOWNLOAD_BYTES + 10)
    _patch_session(monkeypatch, _FakeResponse(data=big))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(web_tools.searxng_search("q", None, None))
    assert exc_info.value.status_code == 502


def test_search_chunked_oversize_is_502(monkeypatch):
    monkeypatch.setattr(tools_config, "TOOLS_CONFIG", _searxng_cfg())
    big = b'{"results": []}' + b" " * (MAX_DOWNLOAD_BYTES + 65_536)
    response = _FakeResponse(data=big)
    response.content = _ChunkedContent(big)
    _patch_session(monkeypatch, response)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(web_tools.searxng_search("q", None, None))
    assert exc_info.value.status_code == 502
    assert "too large" in exc_info.value.detail


def test_search_non_200_status_is_502(monkeypatch):
    monkeypatch.setattr(tools_config, "TOOLS_CONFIG", _searxng_cfg())
    _patch_session(monkeypatch, _FakeResponse(status=302))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(web_tools.searxng_search("q", None, None))
    assert exc_info.value.status_code == 502


def test_search_results_dict_and_number_give_empty(monkeypatch):
    monkeypatch.setattr(tools_config, "TOOLS_CONFIG", _searxng_cfg())
    _patch_session(monkeypatch, _FakeResponse(data=json.dumps({"results": {"a": 1}}).encode()))
    response = asyncio.run(web_tools.searxng_search("q", None, None))
    assert response.results == []
    _patch_session(monkeypatch, _FakeResponse(data=json.dumps({"results": 5}).encode()))
    response = asyncio.run(web_tools.searxng_search("q", None, None))
    assert response.results == []


def test_search_client_error_is_502(monkeypatch):
    monkeypatch.setattr(tools_config, "TOOLS_CONFIG", _searxng_cfg())
    _patch_session(monkeypatch, error=aiohttp.ClientConnectionError("boom"))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(web_tools.searxng_search("q", None, None))
    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Could not reach SearXNG."


def test_search_request_rejects_bad_time_range():
    with pytest.raises(ValidationError):
        WebSearchRequest(query="q", time_range="hour")


def test_search_request_rejects_bad_categories():
    with pytest.raises(ValidationError):
        WebSearchRequest(query="q", categories="general!;rm -rf")


def test_search_request_accepts_social_media_category():
    assert WebSearchRequest(query="q", categories="social media").categories == "social media"


def test_fetch_reports_timeout_not_unreachable(monkeypatch):
    monkeypatch.setattr(tools_config, "TOOLS_CONFIG", _fetch_cfg())
    _patch_session(monkeypatch, error=aiohttp.ServerTimeoutError())
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(web_tools.fetch_page("http://127.0.0.1/x", None))
    assert exc_info.value.status_code == 502
    assert "timed out" in exc_info.value.detail


def test_search_disabled_503(monkeypatch):
    monkeypatch.setattr(tools_config, "TOOLS_CONFIG", _searxng_cfg(enabled=False))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(web_tools.searxng_search("q", None, None))
    assert exc_info.value.status_code == 503


def test_search_missing_url_503(monkeypatch):
    monkeypatch.setattr(tools_config, "TOOLS_CONFIG", _searxng_cfg(url=None))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(web_tools.searxng_search("q", None, None))
    assert exc_info.value.status_code == 503


def test_fetch_disabled_503(monkeypatch):
    monkeypatch.setattr(tools_config, "TOOLS_CONFIG", _fetch_cfg(enabled=False))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(web_tools.fetch_page("http://example.com/x", None))
    assert exc_info.value.status_code == 503


def test_fetch_redirect_is_502(monkeypatch):
    monkeypatch.setattr(tools_config, "TOOLS_CONFIG", _fetch_cfg())
    response = _FakeResponse(status=301, headers={"Location": "http://evil/x"})
    _patch_session(monkeypatch, response)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(web_tools.fetch_page("http://127.0.0.1/x", None))
    assert exc_info.value.status_code == 502
    assert "redirects" in exc_info.value.detail


def test_fetch_non_text_content_is_502(monkeypatch):
    monkeypatch.setattr(tools_config, "TOOLS_CONFIG", _fetch_cfg())
    response = _FakeResponse(data=b"\x00\x01", headers={"Content-Type": "application/octet-stream"})
    _patch_session(monkeypatch, response)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(web_tools.fetch_page("http://127.0.0.1/x", None))
    assert exc_info.value.status_code == 502
    assert "text" in exc_info.value.detail


def test_fetch_too_large_is_502(monkeypatch):
    monkeypatch.setattr(tools_config, "TOOLS_CONFIG", _fetch_cfg())
    big = b"x" * (web_tools.MAX_DOWNLOAD_BYTES + 1)
    _patch_session(monkeypatch, _FakeResponse(data=big, headers={"Content-Type": "text/plain"}))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(web_tools.fetch_page("http://127.0.0.1/x", None))
    assert exc_info.value.status_code == 502
    assert "too large" in exc_info.value.detail


class _ChunkedContent:
    """Mimics aiohttp's StreamReader.read(n): one buffered chunk per call."""

    def __init__(self, data: bytes, chunk_size: int = 65536):
        self._data = data
        self._pos = 0
        self._chunk_size = chunk_size

    async def read(self, limit: int) -> bytes:
        chunk = self._data[self._pos : self._pos + self._chunk_size]
        self._pos += len(chunk)
        return chunk[:limit]


def test_fetch_chunked_oversize_is_502(monkeypatch):
    monkeypatch.setattr(tools_config, "TOOLS_CONFIG", _fetch_cfg())
    big = b"x" * (web_tools.MAX_DOWNLOAD_BYTES + 65_536)
    response = _FakeResponse(data=big, headers={"Content-Type": "text/plain"})
    response.content = _ChunkedContent(big)
    _patch_session(monkeypatch, response)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(web_tools.fetch_page("http://127.0.0.1/x", None))
    assert exc_info.value.status_code == 502
    assert "too large" in exc_info.value.detail


def test_fetch_chunked_body_is_fully_read(monkeypatch):
    monkeypatch.setattr(tools_config, "TOOLS_CONFIG", _fetch_cfg(max_page_chars=300_000))
    body = b"A" * 200_000 + b"MARKER-END"
    response = _FakeResponse(data=body, headers={"Content-Type": "text/plain"})
    response.content = _ChunkedContent(body, chunk_size=50_000)
    _patch_session(monkeypatch, response)
    result = asyncio.run(web_tools.fetch_page("http://127.0.0.1/x", None))
    assert "MARKER-END" in result.text


def test_fetch_json_body_is_pretty_printed(monkeypatch):
    monkeypatch.setattr(tools_config, "TOOLS_CONFIG", _fetch_cfg())
    body = json.dumps({"a": 1, "b": [2, 3]})
    _patch_session(
        monkeypatch, _FakeResponse(data=body.encode(), headers={"Content-Type": "application/json"})
    )
    response = asyncio.run(web_tools.fetch_page("http://127.0.0.1/x", None))
    assert '"a": 1' in response.text
    assert "\n" in response.text


def test_validate_public_url_allows_private_when_configured():
    result = asyncio.run(validate_public_url("http://127.0.0.1:5000/x", allow_private=True))
    assert result == "http://127.0.0.1:5000/x"


def _client(monkeypatch, disable_auth):
    monkeypatch.setattr(auth, "DISABLE_AUTH", disable_auth)
    monkeypatch.setattr(auth, "AUTH_KEYS", auth.AuthKeys(api_key="api", admin_key="admin"))
    return TestClient(setup_app())


def test_tools_status_requires_key(monkeypatch):
    client = _client(monkeypatch, False)
    assert client.get("/v1/tools/status").status_code == 401
    body = client.get("/v1/tools/status", headers={"Authorization": "Bearer api"}).json()
    assert body["web_search"]["enabled"] is False
    assert body["fetch_url"]["enabled"] is False
    assert body["limits"]["max_results"] == 20
    assert "token" not in str(body)


def test_tools_status_open_when_auth_disabled(monkeypatch):
    client = _client(monkeypatch, True)
    response = client.get("/v1/tools/status")
    assert response.status_code == 200


def test_web_search_disabled_by_default(monkeypatch):
    client = _client(monkeypatch, False)
    response = client.post(
        "/v1/tools/web_search",
        headers={"Authorization": "Bearer api"},
        json={"query": "test"},
    )
    assert response.status_code == 503


def test_web_search_endpoint_rejects_bad_time_range(monkeypatch):
    client = _client(monkeypatch, False)
    response = client.post(
        "/v1/tools/web_search",
        headers={"Authorization": "Bearer api"},
        json={"query": "q", "time_range": "hour"},
    )
    assert response.status_code == 422


def test_web_search_endpoint_rejects_long_query(monkeypatch):
    client = _client(monkeypatch, False)
    response = client.post(
        "/v1/tools/web_search",
        headers={"Authorization": "Bearer api"},
        json={"query": "x" * 2001},
    )
    assert response.status_code == 422


def test_fetch_url_disabled_by_default(monkeypatch):
    client = _client(monkeypatch, False)
    response = client.post(
        "/v1/tools/fetch_url",
        headers={"Authorization": "Bearer api"},
        json={"url": "https://example.com"},
    )
    assert response.status_code == 503
