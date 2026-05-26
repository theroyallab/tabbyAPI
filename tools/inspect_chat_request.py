#!/usr/bin/env python3
"""Inspect a TabbyAPI chat completion debug request log."""

import argparse
import json
from pathlib import Path
from typing import Any


def load_debug_request(path: Path) -> dict:
    with path.open("r", encoding="utf8") as log_file:
        payload = json.load(log_file)

    try:
        return payload["request"]
    except KeyError as exc:
        raise SystemExit(f"{path} is not a TabbyAPI request debug log") from exc


def summarize_content(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue

            item_type = item.get("type", "unknown")
            if item_type == "text":
                parts.append(item.get("text", ""))
            elif item_type == "image_url":
                image_url = item.get("image_url", {})
                parts.append(f"[image_url: {image_url.get('url', '')}]")
            else:
                parts.append(f"[{item_type}]")
        return "\n".join(part for part in parts if part)

    return json.dumps(content, ensure_ascii=False)


def create_summary(logged_request: dict) -> dict:
    body = logged_request.get("body", {})
    messages = body.get("messages", [])
    parameters = {key: value for key, value in body.items() if key != "messages"}

    return {
        "request": {
            "id": logged_request.get("id"),
            "url": logged_request.get("url"),
            "path": logged_request.get("path"),
        },
        "parameters": parameters,
        "messages": [
            {
                "index": index,
                "role": message.get("role"),
                "content": summarize_content(message.get("content")),
                "reasoning_content": message.get("reasoning_content"),
                "tool_calls": message.get("tool_calls"),
                "tool_call_id": message.get("tool_call_id"),
            }
            for index, message in enumerate(messages)
        ],
    }


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a logs/debug chat completion request before sharing it."
    )
    parser.add_argument("request_log", type=Path, help="Path to a logs/debug/*.json request log")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the inspection summary as JSON instead of readable text",
    )
    return parser


def print_text_summary(summary: dict) -> None:
    request = summary["request"]
    print(f"Request ID: {request.get('id')}")
    print(f"URL: {request.get('url')}")
    print()

    print("Parameters:")
    for key, value in summary["parameters"].items():
        print(f"  {key}: {json.dumps(value, ensure_ascii=False)}")

    print()
    print("Messages:")
    for message in summary["messages"]:
        print(f"[{message['index']}] role={message.get('role')}")
        content = message.get("content") or ""
        if content:
            print(content)
        if message.get("reasoning_content"):
            print("reasoning_content:")
            print(message["reasoning_content"])
        if message.get("tool_calls"):
            print("tool_calls:")
            print(json.dumps(message["tool_calls"], indent=2, ensure_ascii=False))
        if message.get("tool_call_id"):
            print(f"tool_call_id: {message['tool_call_id']}")
        print()


def main() -> int:
    args = create_argparser().parse_args()
    summary = create_summary(load_debug_request(args.request_log))

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print_text_summary(summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
