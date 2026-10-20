"""The tools config file: optional, validated, watched."""

import asyncio
import contextlib
from pathlib import Path

import pytest
from pydantic import ValidationError
from ruamel.yaml import YAML

from common import tools_config
from common.tools_config import ToolsConfig


def test_defaults_disable_everything():
    cfg = ToolsConfig()
    assert cfg.searxng.enabled is False
    assert cfg.searxng.url is None
    assert cfg.fetch_url.enabled is False
    assert cfg.fetch_url.allow_private_urls is False


def test_sample_file_is_valid():
    yaml = YAML(typ=["rt", "safe"])
    data = yaml.load(Path("tools_config_sample.yml").read_text(encoding="utf8"))
    cfg = ToolsConfig.model_validate(data or {})
    assert cfg.searxng.enabled is False
    assert cfg.searxng.timeout == 10.0
    assert cfg.searxng.max_results == 8
    assert cfg.fetch_url.max_page_chars == 8000


def test_invalid_file_raises(tmp_path):
    bad = tmp_path / "tools_config.yml"
    bad.write_text("searxng: {enabled: notabool}", encoding="utf8")
    with pytest.raises(ValidationError):
        asyncio.run(tools_config._read_tools_file(str(bad)))


def test_empty_file_yields_defaults(tmp_path):
    empty = tmp_path / "tools_config.yml"
    empty.write_text("", encoding="utf8")
    cfg = asyncio.run(tools_config._read_tools_file(str(empty)))
    assert cfg.searxng.enabled is False
    assert cfg.fetch_url.enabled is False
    assert cfg.fetch_url.allow_private_urls is False
    assert cfg.searxng.max_results == 8


def test_custom_values_load(tmp_path):
    cfg_file = tmp_path / "tools_config.yml"
    cfg_file.write_text(
        "searxng: {enabled: true, url: http://127.0.0.1:8080, max_results: 5}\n"
        "fetch_url: {timeout: 7.5}\n",
        encoding="utf8",
    )
    cfg = asyncio.run(tools_config._read_tools_file(str(cfg_file)))
    assert cfg.searxng.enabled is True
    assert cfg.searxng.url == "http://127.0.0.1:8080"
    assert cfg.searxng.max_results == 5
    assert cfg.fetch_url.timeout == 7.5


def test_safe_error_text_redacts_token(tmp_path):
    # A list where a string is expected makes pydantic quote the input value
    # in str(exc); the sanitized text must carry the location and type only.
    bad = tmp_path / "tools_config.yml"
    bad.write_text("searxng: {token: [SUPERSECRET], enabled: true}", encoding="utf8")
    with pytest.raises(ValidationError) as exc_info:
        asyncio.run(tools_config._read_tools_file(str(bad)))

    assert "SUPERSECRET" in str(exc_info.value)

    text = tools_config._safe_error_text(exc_info.value)
    assert "SUPERSECRET" not in text
    assert "('searxng', 'token')" in text
    assert "string_type" in text


def test_watcher_reloads_and_keeps_previous_on_bad_file(monkeypatch, tmp_path):
    watched = tmp_path / "tools_config.yml"
    watched.write_text("searxng: {enabled: false}", encoding="utf8")

    monkeypatch.setattr(tools_config, "TOOLS_FILE", str(watched))
    monkeypatch.setattr(tools_config, "TOOLS_CONFIG", ToolsConfig())
    monkeypatch.setattr(tools_config, "TOOLS_FILE_POLL_INTERVAL", 0.05)

    async def scenario():
        task = asyncio.create_task(tools_config._watch_tools_file())
        try:
            # Let the watcher record the initial mtime before the first write
            await asyncio.sleep(0.1)

            watched.write_text(
                "searxng: {enabled: true, url: http://127.0.0.1:8080}", encoding="utf8"
            )
            await asyncio.sleep(0.3)
            assert tools_config.TOOLS_CONFIG.searxng.enabled is True
            assert tools_config.TOOLS_CONFIG.searxng.url == "http://127.0.0.1:8080"

            watched.write_text("searxng: {enabled: notabool}", encoding="utf8")
            await asyncio.sleep(0.3)
            assert tools_config.TOOLS_CONFIG.searxng.enabled is True
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    asyncio.run(scenario())
