"""Configuration for the built-in web tools (SearXNG search, URL fetch).

Loaded from tools_config.yml at startup and reloaded whenever the file
changes, the same way api_tokens.yml is watched. The file is optional:
without it the web tools stay disabled and everything else works.
"""

import asyncio
import os
from typing import Optional

import aiofiles
from loguru import logger
from pydantic import BaseModel, Field
from ruamel.yaml import YAML

from common.logger import xlogger

TOOLS_FILE = "tools_config.yml"

# Seconds between checks for changes to the tools file
TOOLS_FILE_POLL_INTERVAL = 2.0

# Hard ceilings for per-request overrides sent by clients. An API key holder
# can lower these per request but never raise them past the ceilings.
HARD_MAX_RESULTS = 20
HARD_MAX_RESULT_CHARS = 8000
HARD_MAX_PAGE_CHARS = 32000
HARD_TIMEOUT = 30.0


class SearXNGConfig(BaseModel):
    """Options for the SearXNG search tool."""

    enabled: bool = Field(
        False,
        description=(
            "Enable the web_search tool (default: False). External fetching "
            "stays off until you turn this on, like network.disable_fetch_requests."
        ),
    )
    url: Optional[str] = Field(
        None,
        description=(
            "Base URL of a SearXNG instance, e.g. http://127.0.0.1:8080.\n"
            "The instance must allow the json output format (formats: [html, json]\n"
            "in its settings.yml), and a self-hosted instance is recommended:\n"
            "a public one sees every query the model makes."
        ),
    )
    token: Optional[str] = Field(
        None,
        description=(
            "Optional bearer token for the SearXNG instance. Stays server-side:\n"
            "never sent to clients, never logged, never shown in the UI."
        ),
    )
    timeout: float = Field(
        10.0,
        description="Seconds before a search request is aborted (default: 10).",
        gt=0,
        le=HARD_TIMEOUT,
    )
    max_results: int = Field(
        8,
        description="Maximum results returned to the model per search (default: 8).",
        ge=1,
        le=HARD_MAX_RESULTS,
    )
    max_result_chars: int = Field(
        2000,
        description="Maximum characters per result snippet sent to the model (default: 2000).",
        ge=200,
        le=HARD_MAX_RESULT_CHARS,
    )


class FetchURLConfig(BaseModel):
    """Options for the fetch_url tool."""

    enabled: bool = Field(
        False,
        description=(
            "Enable the fetch_url tool (default: False). It downloads pages by\n"
            "URL; public addresses only unless allow_private_urls is set."
        ),
    )
    timeout: float = Field(
        10.0,
        description="Seconds before a page fetch is aborted (default: 10).",
        gt=0,
        le=HARD_TIMEOUT,
    )
    max_page_chars: int = Field(
        8000,
        description="Maximum characters of extracted page text sent to the model (default: 8000).",
        ge=500,
        le=HARD_MAX_PAGE_CHARS,
    )
    allow_private_urls: bool = Field(
        False,
        description=(
            "Allow fetching private, loopback and link-local addresses "
            "(default: False). Keep it False unless every client of this "
            "instance is fully trusted."
        ),
    )


class ToolsConfig(BaseModel):
    """Top-level tools configuration."""

    searxng: SearXNGConfig = Field(default_factory=SearXNGConfig)
    fetch_url: FetchURLConfig = Field(default_factory=FetchURLConfig)


# The working set of tools options, swapped in as one immutable object on
# reload, like AUTH_KEYS in common/auth.py
TOOLS_CONFIG: ToolsConfig = ToolsConfig()

# Serializes reloads of the tools file
_reload_lock = asyncio.Lock()
_watch_task = None


async def _read_tools_file(path: str = TOOLS_FILE) -> ToolsConfig:
    """Read and validate the tools config file."""

    yaml = YAML(typ=["rt", "safe"])

    async with aiofiles.open(path, "r", encoding="utf8") as tools_file:
        contents = await tools_file.read()
        return ToolsConfig.model_validate(yaml.load(contents) or {})


async def _watch_tools_file():
    """
    Poll the tools file for changes and reload the working set of options.

    A failed reload (partial write, invalid YAML) keeps the previous options.
    Ongoing requests are unaffected either way; a reload only changes which
    options future requests see.
    """

    global TOOLS_CONFIG

    try:
        last_mtime = os.stat(TOOLS_FILE).st_mtime
    except OSError:
        last_mtime = None

    while True:
        await asyncio.sleep(TOOLS_FILE_POLL_INTERVAL)

        try:
            mtime = os.stat(TOOLS_FILE).st_mtime
        except OSError:
            continue

        if mtime == last_mtime:
            continue
        last_mtime = mtime

        async with _reload_lock:
            try:
                TOOLS_CONFIG = await _read_tools_file()
            except Exception as exc:
                xlogger.warning(
                    f"Failed to reload {TOOLS_FILE}, keeping the previous tools config: {exc}"
                )
                continue

        xlogger.info(f"Reloaded tools config from {TOOLS_FILE}.")


async def load_tools_config():
    """Load the tools config from tools_config.yml. The file is optional;
    without it the web tools stay disabled."""

    global TOOLS_CONFIG
    global _watch_task

    try:
        TOOLS_CONFIG = await _read_tools_file()
    except FileNotFoundError:
        TOOLS_CONFIG = ToolsConfig()
        logger.info(
            f"No {TOOLS_FILE} found. The web tools stay disabled; copy "
            "tools_config_sample.yml to tools_config.yml to enable them."
        )
    except Exception as exc:
        xlogger.warning(f"Failed to load {TOOLS_FILE}: {exc}. Web tools stay disabled.")
        TOOLS_CONFIG = ToolsConfig()

    # Reload the options whenever the file changes, so the SearXNG instance
    # can be switched without a server restart
    if _watch_task is None:
        _watch_task = asyncio.create_task(_watch_tools_file())
