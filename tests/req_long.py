import random
import string

import yaml
from _common import *

BASE_URL = "http://localhost:5000/v1"
MODEL = "/mnt/str/models/qwen3.5-35b-a3b/exl3/4.09bpw/"

oai_request = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": "".join(random.choices(string.ascii_letters, k=4))
            + "All work and no play. " * 5000,
        }
    ],
    "max_tokens": 500,
    "stream_options": {"include_usage": True},
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
