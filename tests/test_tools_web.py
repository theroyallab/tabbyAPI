"""The built-in web tools: URL guards, result shaping, and endpoint auth."""

import asyncio

import pytest

from endpoints.OAI.utils.web_tools import (
    _blocked_ip,
    _clamp,
    _clean_snippet,
    _extract_text,
    validate_public_url,
)


def test_blocked_ips():
    for ip in (
        "127.0.0.1", "10.1.2.3", "172.16.0.1", "192.168.1.1",
        "169.254.1.1", "::1", "fe80::1", "fc00::1", "0.0.0.0",
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


def test_extract_text_drops_scripts():
    title, text = _extract_text(
        "<html><head><title>T</title><style>x</style></head>"
        "<body><script>bad()</script><p>hello</p></body></html>"
    )
    assert title == "T"
    assert text == "hello"


def test_validate_public_url_rejects_private():
    with pytest.raises(Exception):
        asyncio.run(validate_public_url("http://127.0.0.1:5000/x"))
    with pytest.raises(Exception):
        asyncio.run(validate_public_url("ftp://example.com/x"))