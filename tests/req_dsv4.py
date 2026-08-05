from _common import *

BASE_URL = "http://localhost:5010/v1"
MODEL = "/mnt/str/models/deepseek-v4-flash-0731/exl3/3.04bpw/"

# DeepSeek-V4's template defaults to thinking mode
plain_request = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "What is 8 * 12 + 5? Answer briefly."}],
    "stream_options": {"include_usage": True},
}

# Chat mode via the top-level boolean
no_think_request = {
    "model": MODEL,
    "enable_thinking": False,
    "messages": [{"role": "user", "content": "What is 8 * 12 + 5? Answer briefly."}],
    "stream_options": {"include_usage": True},
}

tool_request = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": "What's the weather like in Paris right now? Use the tool.",
        }
    ],
    "stream_options": {"include_usage": True},
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "days": {
                            "type": "integer",
                            "description": "Forecast days ahead (0 = current)",
                        },
                    },
                    "required": ["location"],
                },
            },
        },
    ],
    "tool_choice": "auto",
}

# Parallel tool calls answered OUT OF ORDER: Tabby must reorder the tool
# messages to match the tool_calls order before templating
tool_response_request = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": "What's the weather in Paris and Tokyo right now? Use the tool for both.",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_paris_0000000000000000",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}',
                    },
                },
                {
                    "id": "call_tokyo_0000000000000000",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Tokyo"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_tokyo_0000000000000000",
            "content": '{"temperature": "31C", "conditions": "humid"}',
        },
        {
            "role": "tool",
            "tool_call_id": "call_paris_0000000000000000",
            "content": '{"temperature": "21C", "conditions": "light rain"}',
        },
    ],
    "tools": tool_request["tools"],
    "tool_choice": "auto",
}


def main():
    _, api_key = load_api_keys()

    test_chat_request(api_key, BASE_URL, plain_request.copy(), n=1)
    test_chat_request(api_key, BASE_URL, no_think_request.copy(), n=1)
    test_chat_streaming(api_key, BASE_URL, plain_request.copy(), n=1)
    test_chat_request(api_key, BASE_URL, tool_request.copy(), n=1)
    test_chat_streaming(api_key, BASE_URL, tool_request.copy(), n=1)
    test_chat_request(api_key, BASE_URL, tool_response_request.copy(), n=1)


if __name__ == "__main__":
    main()
