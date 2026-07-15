"""Utilities for writing reproducible request debug logs."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiofiles
from fastapi import Request

from common.logger import xlogger
from common.tabby_config import config


DEBUG_LOG_DIR = Path("logs/debug")

SENSITIVE_HEADERS = {
    "authorization",
    "cookie",
    "host",
    "set-cookie",
    "user-agent",
    "x-admin-key",
    "x-api-key",
}

SENSITIVE_KEYS = {
    "admin_key",
    "api_key",
    "authorization",
    "hf_token",
    "token",
    "x-admin-key",
    "x-api-key",
}


def redact_sensitive_values(value: Any) -> Any:
    """Recursively redact auth-like fields from a JSON-compatible object."""

    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if key.lower() in SENSITIVE_KEYS:
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = redact_sensitive_values(item)
        return redacted

    if isinstance(value, list):
        return [redact_sensitive_values(item) for item in value]

    return value


def redact_headers(headers) -> dict:
    """Return request headers with credential-bearing values removed."""

    return {
        key: "[REDACTED]" if key.lower() in SENSITIVE_HEADERS else value
        for key, value in headers.items()
    }


async def write_chat_completion_request_log(request: Request, body: dict) -> Path | None:
    """Write a /v1/chat/completions request as a JSON debug artifact."""

    timestamp = datetime.now(timezone.utc)
    request_id = getattr(request.state, "id", "unknown")
    filename = f"{timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}_{request_id}.json"
    log_path = DEBUG_LOG_DIR / filename

    payload = {
        "schema": "tabbyapi.chat_completion_request_log.v1",
        "created_at": timestamp.isoformat(),
        "request": {
            "id": "[REDACTED]",
            "method": request.method,
            "url": f"[REDACTED]{request.url.path}",
            "path": request.url.path,
            "query": redact_sensitive_values(dict(request.query_params)),
            "headers": redact_headers(request.headers),
            "body": redact_sensitive_values(body),
        },
    }

    try:
        DEBUG_LOG_DIR.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(log_path, "w", encoding="utf8") as debug_file:
            await debug_file.write(json.dumps(payload, indent=2, ensure_ascii=False))
            await debug_file.write("\n")
    except Exception as exc:
        xlogger.error(
            "Failed to write chat completion request debug log",
            str(exc),
            details=f"\n{exc}",
        )
        return None

    xlogger.info(f"Wrote chat completion request debug log: {log_path}")
    return log_path


async def log_chat_completion_request(request: Request):
    """FastAPI dependency to log chat completion request bodies when enabled."""

    if not config.logging.log_chat_completion_requests:
        return

    try:
        body = await request.json()
    except Exception:
        body = {"_invalid_json_body": "[omitted: request body was not valid JSON]"}

    await write_chat_completion_request_log(request, body)
