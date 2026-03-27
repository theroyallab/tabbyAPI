"""
Internal logging utility.
"""

import logging
import os
import requests
import json
from datetime import datetime, timezone

from loguru import logger
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from common.utils import unwrap

_w = os.getenv("TABBY_LOG_CONSOLE_WIDTH")
_default_console_width = int(_w) if _w is not None and _w.isnumeric() else None
RICH_CONSOLE = Console(width=_default_console_width)
LOG_LEVEL = os.getenv("TABBY_LOG_LEVEL", "INFO")


def get_progress_bar():
    return Progress(console=RICH_CONSOLE)


def get_loading_progress_bar():
    """Gets a pre-made progress bar for loading tasks."""

    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=RICH_CONSOLE,
    )


def _log_formatter(record: dict):
    """Log message formatter."""

    color_map = {
        "TRACE": "dim blue",
        "DEBUG": "cyan",
        "INFO": "green",
        "SUCCESS": "bold green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold white on red",
    }

    time = record.get("time")
    colored_time = f"[grey37]{time:YYYY-MM-DD HH:mm:ss.SSS}[/grey37]"

    level = record.get("level")
    level_color = color_map.get(level.name, "cyan")
    colored_level = f"[{level_color}]{level.name}[/{level_color}]:"

    separator = " " * (9 - len(level.name))

    message = unwrap(record.get("message"), "")

    # Replace once loguru allows for turning off str.format
    message = message.replace("{", "{{").replace("}", "}}").replace("<", "\<")

    # Escape markup tags from Rich
    message = escape(message)
    lines = message.splitlines()

    fmt = ""
    if len(lines) > 1:
        fmt = "\n".join(
            [f"{colored_time} {colored_level}{separator}{line}" for line in lines]
        )
    else:
        fmt = f"{colored_time} {colored_level}{separator}{message}"

    return fmt


# Uvicorn log handler
# Uvicorn log portions inspired from https://github.com/encode/uvicorn/discussions/2027#discussioncomment-6432362
class UvicornLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        logger.opt(exception=record.exc_info).log(
            record.levelname, self.format(record).rstrip()
        )


# Uvicorn config for logging. Passed into run when creating all loggers in server
UVICORN_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "uvicorn": {
            "class": f"{UvicornLoggingHandler.__module__}.{UvicornLoggingHandler.__qualname__}",  # noqa
        },
    },
    "root": {"handlers": ["uvicorn"], "propagate": False, "level": LOG_LEVEL},
}


def setup_logger():
    """Bootstrap the logger."""

    logger.remove()

    logger.add(
        RICH_CONSOLE.print,
        level=LOG_LEVEL,
        format=_log_formatter,
        colorize=True,
    )
    # Add file logging
    logger.add(
        "logs/{time}.log",
        level=LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        rotation="20 MB",  # Rotate file when it reaches 20MB
        retention="1 week",  # Keep logs for 1 week
        compression="zip",  # Compress rotated log
    )


"""
Extended logging via Seq.
"""


class XLogger:
    def __init__(self):
        self.seqlog_url = None
        self.headers = {}
        self.enabled = False

    def _get_timestamp_now(self):
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def setup(
        self, seqlog_url: str = "http://localhost:5341", api_key: str | None = None
    ):
        self.seqlog_url = seqlog_url.rstrip("/")
        self.headers = {"Content-Type": "application/vnd.serilog.clef"}
        if api_key:
            self.headers["X-Seq-ApiKey"] = api_key

        # Check if seqlog is reachable
        try:
            r = requests.post(
                self.seqlog_url + "/ingest/clef",
                data=(
                    f'{{"@t":"{self._get_timestamp_now()}",'
                    f'"@m":"TabbyAPI startup probe"}}\n'
                ),
                headers=self.headers,
                timeout=2,
            )
            r.raise_for_status()
        except requests.RequestException as e:
            logger.info(
                f"Failed to initialize seqlog handler for server at "
                f"{self.seqlog_url}: {e}"
                f"seqlog logging is disabled."
            )
            return

        self.enabled = True
        logger.info(f"Enabled logging to seqlog instance at {self.seqlog_url}")

    def _commit(self, log_level: str, log_message: str, log_extra: dict):
        if not self.enabled:
            return

        try:
            if log_extra is None:
                log_extra = {}
            elif not isinstance(log_extra, dict):
                log_extra = {"extra": str(log_extra)}
            event = {
                "@t": self._get_timestamp_now(),
                "@m": log_message,
                "@l": log_level,
                **log_extra,
            }
            try:
                data = json.dumps(event, default=str) + "\n"
            except Exception as e:
                data = "## Failed to serialize log data: " + str(e)
            r = requests.post(
                self.seqlog_url + "/ingest/clef",
                data=data,
                headers=self.headers,
                timeout=2,
            )
            r.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"Failed to write log event to Seq, logging disabled: {e}")
            self.enabled = False

    def _compose(self, log_message, details):
        return (log_message + " " + details) if details else log_message

    def verbose(
        self,
        log_message: str,
        log_extra: dict | None = None,
        details: str | None = None,
    ):
        self._commit("Verbose", log_message, log_extra)

    def debug(
        self,
        log_message: str,
        log_extra: dict | None = None,
        details: str | None = None,
    ):
        logger.debug(self._compose(log_message, details))
        self._commit("Debug", log_message, log_extra)

    def info(
        self,
        log_message: str,
        log_extra: dict | None = None,
        details: str | None = None,
    ):
        logger.info(self._compose(log_message, details))
        self._commit("Information", log_message, log_extra)

    def warning(
        self,
        log_message: str,
        log_extra: dict | None = None,
        details: str | None = None,
    ):
        logger.warning(self._compose(log_message, details))
        self._commit("Warning", log_message, log_extra)

    def error(
        self,
        log_message: str,
        log_extra: dict | None = None,
        details: str | None = None,
    ):
        logger.error(self._compose(log_message, details))
        self._commit("Error", log_message, log_extra)


xlogger = XLogger()
