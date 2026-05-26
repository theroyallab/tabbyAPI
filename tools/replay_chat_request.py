#!/usr/bin/env python3
"""Replay a TabbyAPI chat completion debug request log."""

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

col_default = "\u001b[0m"
col_yellow = "\u001b[33;1m"


def load_debug_request(path: Path) -> dict:
    with path.open("r", encoding="utf8") as log_file:
        payload = json.load(log_file)

    try:
        return payload["request"]
    except KeyError as exc:
        raise SystemExit(f"{path} is not a TabbyAPI request debug log") from exc


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay a logs/debug chat completion request and stream raw output."
    )
    parser.add_argument("request_log", type=Path, help="Path to a logs/debug/*.json request log")
    parser.add_argument(
        "--url",
        help=(
            "Endpoint URL to replay against. Defaults to the saved URL with its path replaced "
            "by /v1/chat/completions."
        ),
    )
    parser.add_argument("--api-key", help="API or admin key to send with request")
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Request timeout in seconds. Defaults to no explicit timeout.",
    )
    parser.add_argument(
        "--format",
        action="store_true",
        help="After replay completes, print parsed assistant text from the response.",
    )
    return parser


def response_to_text(response_text: str) -> str:
    """Extract assistant text from a non-streaming or streaming chat response."""

    stripped = response_text.strip()
    if not stripped:
        return ""

    parts_r, parts_c, parts_t = [], [], []

    if stripped.startswith("{"):
        payload = json.loads(stripped)
        for choice in payload.get("choices", []):
            message = choice.get("message", {})
            if message.get("reasoning_content"):
                parts_r.append(message["reasoning_content"])
            if message.get("content"):
                parts_c.append(message["content"])
            if message.get("tool_calls"):
                parts_t.append(json.dumps(message["tool_calls"], indent=2, ensure_ascii=False))

    else:
        for line in response_text.splitlines():
            line = line.strip()
            if not line or line == "[DONE]":
                continue
            if line.startswith("data:"):
                line = line.removeprefix("data:").strip()
            if line == "[DONE]":
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            for choice in payload.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("reasoning_content"):
                    parts_r.append(delta["reasoning_content"])
                if delta.get("content"):
                    parts_c.append(delta["content"])
                if delta.get("tool_calls"):
                    parts_t.append(json.dumps(delta["tool_calls"], ensure_ascii=False))

    return (
        f"\n\n{col_yellow} --- reasoniong_content ---{col_default}\n\n"
        + "".join(parts_r)
        + f"\n\n{col_yellow} --- content ---{col_default}\n\n"
        + "".join(parts_c)
        + f"\n\n{col_yellow} --- tool_calls ---{col_default}\n\n"
        + "".join(parts_t)
    )


def main() -> int:
    args = create_argparser().parse_args()
    logged_request = load_debug_request(args.request_log)
    body = logged_request.get("body")
    if not isinstance(body, dict):
        raise SystemExit("Debug log does not contain a JSON request body")

    url = args.url
    if not url:
        saved_url = logged_request.get("url", "http://127.0.0.1:5000/v1/chat/completions")
        url = saved_url.split("/v1/chat/completions", 1)[0] + "/v1/chat/completions"

    headers = {
        "accept": logged_request.get("headers", {}).get("accept", "*/*"),
        "content-type": "application/json",
    }
    if args.api_key:
        headers["x-api-key"] = args.api_key

    request = Request(
        url,
        data=json.dumps(body).encode("utf8"),
        headers=headers,
        method="POST",
    )

    collected = bytearray()
    try:
        with urlopen(request, timeout=args.timeout) as response:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                collected.extend(chunk)
    except HTTPError as exc:
        sys.stderr.write(f"HTTP {exc.code} {exc.reason}\n")
        sys.stderr.buffer.write(exc.read())
        return 1
    except URLError as exc:
        sys.stderr.write(f"Request failed: {exc.reason}\n")
        return 1

    if args.format:
        response_text = collected.decode("utf8", errors="replace")
        formatted = response_to_text(response_text)
        print("\n\n--- Parsed assistant text ---")
        print(formatted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
