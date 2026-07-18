from _common import *

BASE_URL = "http://localhost:5010/v1"
MODEL = "/mnt/str/models/gpt-oss-20b/exl3/4.00bpw/"

plain_request = {
    "model": MODEL,
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
    test_chat_streaming(api_key, BASE_URL, plain_request.copy(), n=1)
    test_chat_request(api_key, BASE_URL, tool_request.copy(), n=1)
    test_chat_streaming(api_key, BASE_URL, tool_request.copy(), n=1)
    test_chat_request(api_key, BASE_URL, tool_response_request.copy(), n=1)


if __name__ == "__main__":
    main()
