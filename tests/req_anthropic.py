"""
Manual checks for the Anthropic Messages API against a running server.

Enable the API first, in config.yml:

    network:
      api_servers: ["OAI", "Anthropic"]

The requests are sent with the Anthropic header and auth conventions
(x-api-key, anthropic-version) so this also exercises the header path real
SDK clients use.
"""

import json
from pprint import pprint

import httpx

from _common import load_api_keys

BASE_URL = "http://localhost:5000/v1"
MODEL = "/mnt/str/models/qwen3.5-35b-a3b/exl3/4.09bpw/"

simple_request = {
    "model": MODEL,
    "max_tokens": 512,
    "system": "You are a concise assistant.",
    "messages": [{"role": "user", "content": "Name three primary colors."}],
}

block_request = {
    "model": MODEL,
    "max_tokens": 512,
    # Clients routinely split the system prompt across blocks
    "system": [
        {
            "type": "text",
            "text": "You are a concise assistant.",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "<env>The user is testing an API shim.</env>"},
    ],
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "Say hello."}]},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "A greeting was requested.", "signature": "x"},
                {"type": "text", "text": "Hello!"},
            ],
        },
        {"role": "user", "content": "Now say goodbye."},
    ],
}

stop_sequence_request = {
    "model": MODEL,
    "max_tokens": 512,
    "stop_sequences": ["END"],
    "messages": [
        {
            "role": "user",
            "content": "Count from 1 to 10, one number per line, then write END.",
        }
    ],
}


def post(api_key, path, request):
    return httpx.post(
        f"{BASE_URL}{path}",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        json=request,
        timeout=300,
    )


def test_message(api_key, request, label):
    print("\n\n")
    print("-" * 80)
    print(f"MESSAGES REQUEST: {label}")
    print("-" * 80)

    data = post(api_key, "/messages", request).json()
    pprint(data, width=160)

    if data.get("type") == "error":
        return data

    for block in data.get("content", []):
        if block["type"] == "thinking":
            print(f"\n[thinking]\n{block['thinking']}")
        elif block["type"] == "text":
            print(f"\n[text]\n{block['text']}")

    print(f"\nStop reason: {data.get('stop_reason')} (sequence: {data.get('stop_sequence')})")
    print(f"Usage: {data.get('usage')}")

    return data


def test_message_streaming(api_key, request, label):
    print("\n\n")
    print("-" * 80)
    print(f"STREAMING MESSAGES REQUEST: {label}")
    print("-" * 80)

    request = {**request, "stream": True}

    # Accumulate the way an SDK does, so a malformed block lifecycle shows up
    blocks = {}
    order = []
    final = {}
    event_names = []

    with httpx.stream(
        "POST",
        f"{BASE_URL}/messages",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        json=request,
        timeout=300,
    ) as response:
        name = None
        for line in response.iter_lines():
            line = line.rstrip("\r")
            if line.startswith("event:"):
                name = line[len("event:") :].strip()
                event_names.append(name)
                continue
            if not line.startswith("data:"):
                continue

            payload = json.loads(line[len("data:") :].strip())

            if name == "content_block_start":
                blocks[payload["index"]] = dict(payload["content_block"])
                order.append(payload["index"])
                kind = payload["content_block"]["type"]
                print(f"\n\n[{kind}][{payload['index']}]")
            elif name == "content_block_delta":
                delta = payload["delta"]
                key = "thinking" if delta["type"] == "thinking_delta" else "text"
                blocks[payload["index"]][key] += delta[key]
                print(delta[key], end="", flush=True)
            elif name == "message_delta":
                final = payload
            elif name == "error":
                print(f"\n\n[error] {payload}")

    print(f"\n\nEvent order: {event_names}")
    print(f"Accumulated blocks: {[blocks[i] for i in order]}")
    print(f"Stop: {final.get('delta')}")
    print(f"Usage: {final.get('usage')}")

    return blocks


def test_count_tokens(api_key, request, label):
    print("\n\n")
    print("-" * 80)
    print(f"COUNT TOKENS REQUEST: {label}")
    print("-" * 80)

    counted = {key: request[key] for key in ("model", "system", "messages") if key in request}
    data = post(api_key, "/messages/count_tokens", counted).json()
    pprint(data, width=160)

    return data


def main():
    api_key, _ = load_api_keys()

    test_message(api_key, simple_request.copy(), "plain text")
    test_message(api_key, block_request.copy(), "content blocks and replayed thinking")
    test_message(api_key, stop_sequence_request.copy(), "client stop sequence")

    test_message_streaming(api_key, simple_request.copy(), "plain text")
    test_message_streaming(api_key, block_request.copy(), "content blocks")
    test_message_streaming(api_key, stop_sequence_request.copy(), "client stop sequence")

    test_count_tokens(api_key, simple_request, "plain text")
    test_count_tokens(api_key, block_request, "content blocks")

    # Unsupported blocks must fail loudly rather than drop conversation
    unsupported = simple_request.copy()
    unsupported["messages"] = [
        {"role": "user", "content": [{"type": "image", "source": {"type": "base64"}}]}
    ]
    test_message(api_key, unsupported, "unsupported block (expects an error envelope)")


if __name__ == "__main__":
    main()
