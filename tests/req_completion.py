import random
import string

import yaml
from _common import *

BASE_URL = "http://localhost:5000/v1"
MODEL = "/mnt/str/models/qwen3.5-35b-a3b/exl3/4.09bpw/"

oai_request = {
    "model": MODEL,
    "prompt": "".join(random.choices(string.ascii_letters, k=4)) + " All work and no play." * 50,
    "max_tokens": 10,
    "stream_options": {"include_usage": True},
}

oai_request_logprobs = {
    "model": MODEL,
    "prompt": "".join(random.choices(string.ascii_letters, k=4)) + " All work and no play." * 50,
    "max_tokens": 10,
    "stream_options": {"include_usage": True},
    "logprobs": 3,
}

oai_request_batch = {
    "model": MODEL,
    "prompt": [
        "".join(random.choices(string.ascii_letters, k=4)) + " All work and no play." * 50,
        "1 2 3 4 5 6 7 8 9 10 11 12",
    ],
    "max_tokens": 50,
    "stream_options": {"include_usage": True},
}

oai_request_long = {
    "model": MODEL,
    "prompt": [
        "".join(random.choices(string.ascii_letters, k=4)) + " All work and no play." * 500,
        "1 2 3 4 5 6 7 8 9 10 11 12" * 200,
    ],
    "max_tokens": 1000,
    "stream_options": {"include_usage": False},
}

oai_request_long_s = {
    "model": MODEL,
    "prompt": [
        "".join(random.choices(string.ascii_letters, k=4)) + " All work and no play." * 500,
        "1 2 3 4 5 6 7 8 9 10 11 12" * 200,
    ],
    "max_tokens": 1000,
    "stream_options": {"include_usage": True},
}


def main():
    with open("api_tokens.yml") as f:
        tokens = yaml.safe_load(f)
        api_key = tokens["admin_key"]

    # test_comp_request(api_key, BASE_URL, oai_request_long.copy(), n=1)
    # test_comp_streaming(api_key, BASE_URL, oai_request_long_s.copy(), n=1)
    test_comp_streaming(api_key, BASE_URL, oai_request_logprobs.copy(), n=2, rawdump=True)
    test_comp_request(api_key, BASE_URL, oai_request.copy(), n=1)
    test_comp_request(api_key, BASE_URL, oai_request_logprobs.copy(), n=2)
    test_comp_request(api_key, BASE_URL, oai_request.copy(), n=4)
    test_comp_request(api_key, BASE_URL, oai_request_batch.copy(), n=2)
    test_comp_streaming(api_key, BASE_URL, oai_request.copy(), n=1)
    test_comp_streaming(api_key, BASE_URL, oai_request.copy(), n=2, rawdump=True)
    test_comp_streaming(api_key, BASE_URL, oai_request.copy(), n=4)


if __name__ == "__main__":
    main()
