import yaml
from _common import *

BASE_URL = "http://localhost:5000/v1"
MODEL = "/mnt/str/models/qwen3.5-35b-a3b/exl3/4.09bpw/"

oai_request = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": "What's the weather like in Paris and London? Also pick "
            "some third, random city in California (with less than 100k "
            "inhabitants) and encrypt the name using sznorfleq.",
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
        {
            "type": "function",
            "function": {
                "name": "sznorfleq_encode",
                "description": "Encode a string with sznorfleq encryption.",
                "parameters": {
                    "type": "object",
                    "properties": {"input": {"type": "string", "description": "Input string"}},
                    "required": ["input"],
                },
            },
        },
    ],
    "tool_choice": "auto",
}

non_tool_request = {
    "model": MODEL,
    "template_vars": {
        "enable_thinking": False,
    },
    "messages": [{"role": "user", "content": "Hello."}],
}


def main():
    with open("api_tokens.yml") as f:
        tokens = yaml.safe_load(f)
        api_key = tokens["admin_key"]

    test_chat_streaming(api_key, BASE_URL, non_tool_request.copy(), n=1)
    test_chat_request(api_key, BASE_URL, oai_request.copy(), n=1)
    test_chat_request(api_key, BASE_URL, oai_request.copy(), n=2)
    test_chat_request(api_key, BASE_URL, oai_request.copy(), n=4)
    test_chat_streaming(api_key, BASE_URL, oai_request.copy(), n=1)
    test_chat_streaming(api_key, BASE_URL, oai_request.copy(), n=2)
    test_chat_streaming(api_key, BASE_URL, oai_request.copy(), n=4)


if __name__ == "__main__":
    main()
