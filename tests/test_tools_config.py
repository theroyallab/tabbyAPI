"""The tools config file: optional, validated, watched."""

import asyncio
from pathlib import Path

import pytest
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
    with pytest.raises(Exception):
        asyncio.run(tools_config._read_tools_file(str(bad)))


def test_empty_file_yields_defaults(tmp_path):
    empty = tmp_path / "tools_config.yml"
    empty.write_text("", encoding="utf8")
    cfg = asyncio.run(tools_config._read_tools_file(str(empty)))
    assert cfg.searxng.enabled is False
