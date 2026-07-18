from _common import *

BASE_URL = "http://localhost:5010/v1"
MODEL = "/mnt/str/models/hy3/exl3/3.05bpw/"

plain_request = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "What is 8 * 12 + 5? Answer briefly."}],
    "stream_options": {"include_usage": True},
}

# Hy3's template defaults to reasoning_effort: no_think; enable reasoning
# via the top-level request field to exercise the reasoning parser
reasoning_request = {
    "model": MODEL,
    "reasoning_effort": "low",
    "messages": [{"role": "user", "content": "What is 8 * 12 + 5? Answer briefly."}],
    "stream_options": {"include_usage": True},
}

# Client override back to no reasoning (relevant when the model config sets
# template_vars_default: {reasoning_effort: high})
no_think_request = {
    "model": MODEL,
    "reasoning_effort": "no_think",
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
                    "properties": {"location": {"type": "string", "description": "City name"}},
                    "required": ["location"],
                },
            },
        },
    ],
    "tool_choice": "auto",
}

# Continue the conversation after a tool response
tool_response_request = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": "What's the weather like in Paris right now? Use the tool.",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_0123456789abcdef01234567",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": "21C", "conditions": "light rain"}',
        },
    ],
    "tools": tool_request["tools"],
    "tool_choice": "auto",
}


def main():
    _, api_key = load_api_keys()

    test_chat_request(api_key, BASE_URL, plain_request.copy(), n=1)
    test_chat_request(api_key, BASE_URL, reasoning_request.copy(), n=1)
    test_chat_request(api_key, BASE_URL, no_think_request.copy(), n=1)
    test_chat_streaming(api_key, BASE_URL, reasoning_request.copy(), n=1)
    test_chat_request(api_key, BASE_URL, tool_request.copy(), n=1)
    test_chat_streaming(api_key, BASE_URL, tool_request.copy(), n=1)
    test_chat_request(api_key, BASE_URL, tool_response_request.copy(), n=1)


if __name__ == "__main__":
    main()
