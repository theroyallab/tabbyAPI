import re

import yaml
from _common import *

BASE_URL = "http://localhost:5000/v1"
MODEL = "/mnt/str/models/qwen3.5-9b/exl3/5.00bpw_mul1/"

PHONE_REGEX = r"\d{3}-\d{4}"
DATE_REGEX = r"\d{4}-\d{2}-\d{2}"

YESNO_KBNF = 'start ::= "yes" | "no";'
SENTENCE_KBNF = (
    'start ::= greeting " " subject "!";\n'
    'greeting ::= "Hello" | "Goodbye";\n'
    'subject ::= "world" | "friend";\n'
)

comp_request_phone = {
    "model": MODEL,
    "prompt": "My phone number is ",
    "max_tokens": 20,
    "regex_pattern": PHONE_REGEX,
}

comp_request_date = {
    "model": MODEL,
    "prompt": "The date today is ",
    "max_tokens": 20,
    "regex_pattern": DATE_REGEX,
}

chat_request_date = {
    "model": MODEL,
    "template_vars": {
        "enable_thinking": False,
    },
    "messages": [
        {
            "role": "user",
            "content": "What is the date of the summer solstice in 2026? Reply with only the date.",
        }
    ],
    "max_tokens": 20,
    "regex_pattern": DATE_REGEX,
}

comp_request_yesno = {
    "model": MODEL,
    "prompt": "Question: Is the sky blue?\nAnswer:",
    "max_tokens": 10,
    "grammar_string": YESNO_KBNF,
}

comp_request_sentence = {
    "model": MODEL,
    "prompt": "A friendly greeting:",
    "max_tokens": 20,
    "grammar_string": SENTENCE_KBNF,
}

failures = []


def check(label, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {label}")
    if not condition:
        failures.append(label)


def main():
    with open("api_tokens.yml") as f:
        tokens = yaml.safe_load(f)
        api_key = tokens["admin_key"]

    # Regex on the completions endpoint
    data = test_comp_request(api_key, BASE_URL, comp_request_phone.copy(), n=1)
    check("regex phone: output matches pattern", re.fullmatch(PHONE_REGEX, data["choices"][0]["text"].strip()))

    data = test_comp_request(api_key, BASE_URL, comp_request_date.copy(), n=1)
    check("regex date: output matches pattern", re.fullmatch(DATE_REGEX, data["choices"][0]["text"].strip()))

    # Regex, n=2 (filters must be independent per generation)
    data = test_comp_request(api_key, BASE_URL, comp_request_phone.copy(), n=2)
    for idx, choice in enumerate(data["choices"]):
        check(f"regex phone n=2 choice {idx}: output matches pattern", re.fullmatch(PHONE_REGEX, choice["text"].strip()))

    # Regex on the chat completions endpoint
    data = test_chat_request(api_key, BASE_URL, chat_request_date.copy(), n=1)
    check("regex chat date: output matches pattern", re.fullmatch(DATE_REGEX, data["choices"][0]["message"]["content"].strip()))

    # KBNF grammars on the completions endpoint
    data = test_comp_request(api_key, BASE_URL, comp_request_yesno.copy(), n=1)
    check("kbnf yes/no: output conforms", data["choices"][0]["text"].strip() in {"yes", "no"})

    data = test_comp_request(api_key, BASE_URL, comp_request_sentence.copy(), n=1)
    check(
        "kbnf sentence: output conforms",
        re.fullmatch(r"(Hello|Goodbye) (world|friend)!", data["choices"][0]["text"].strip()),
    )

    print()
    if failures:
        print(f"{len(failures)} FAILED:")
        for f_ in failures:
            print(f"  - {f_}")
        exit(1)
    else:
        print("All grammar checks passed")


if __name__ == "__main__":
    main()
