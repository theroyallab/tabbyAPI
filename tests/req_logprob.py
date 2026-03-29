import yaml
from _common import *

BASE_URL = "http://localhost:5000/v1"
MODEL = "/mnt/str/models/qwen3.5-35b-a3b/exl3/4.09bpw/"

oai_request = {
    "model": MODEL,
    "template_vars": {
        "enable_thinking": False,
    },
    "messages": [
        {
            "role": "user",
            "content": "Write a Haiku about fish, and start each line with a fish-related emoji.",
        }
    ],
    "logprobs": True,
    "top_logprobs": 7,
    "stream_options": {"include_usage": True},
    # "max_tokens": 200,
}

oai_request_2 = {
    "model": MODEL,
    "template_vars": {
        "enable_thinking": True,
    },
    "messages": [{"role": "user", "content": "What is the mass of a water molecule, in kg?"}],
    "logprobs": True,
    "top_logprobs": 5,
    "stream_options": {"include_usage": True},
    # "max_tokens": 200,
}


def main():
    with open("api_tokens.yml") as f:
        tokens = yaml.safe_load(f)
        api_key = tokens["admin_key"]

    test_chat_request(api_key, BASE_URL, oai_request.copy(), n=1)
    test_chat_request(api_key, BASE_URL, oai_request.copy(), n=2)
    test_chat_streaming(api_key, BASE_URL, oai_request.copy(), n=1)
    test_chat_streaming(api_key, BASE_URL, oai_request.copy(), n=2)
    #
    test_chat_streaming(api_key, BASE_URL, oai_request.copy(), n=1, rawdump=True)
    test_chat_request(api_key, BASE_URL, oai_request_2.copy(), n=1)


if __name__ == "__main__":
    main()
