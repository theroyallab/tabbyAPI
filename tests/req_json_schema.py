import json

from _common import *

BASE_URL = "http://localhost:5000/v1"
MODEL = "/mnt/str/models/qwen3.5-9b/exl3/5.00bpw_mul1/"

PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "hobbies": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "age", "hobbies"],
}

LIST_SCHEMA = {
    "type": "array",
    "items": {"type": "string"},
    "minItems": 3,
}

comp_request = {
    "model": MODEL,
    "prompt": "Generate a random person as JSON:",
    "max_tokens": 200,
    "json_schema": PERSON_SCHEMA,
}

comp_request_array = {
    "model": MODEL,
    "prompt": "List some fruits as a JSON array:",
    "max_tokens": 200,
    "json_schema": LIST_SCHEMA,
}

chat_request = {
    "model": MODEL,
    "template_vars": {
        "enable_thinking": False,
    },
    "messages": [
        {
            "role": "user",
            "content": "Make up a person who enjoys fishing. Respond in JSON.",
        }
    ],
    "max_tokens": 200,
    "json_schema": PERSON_SCHEMA,
}

failures = []


def check(label, condition):
    print(f"[{'PASS' if condition else 'FAIL'}] {label}")
    if not condition:
        failures.append(label)


def validate_person(label, text):
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        check(f"{label}: output is valid JSON", False)
        return

    check(f"{label}: output is valid JSON", True)
    check(
        f"{label}: object has required fields",
        isinstance(obj, dict) and all(k in obj for k in PERSON_SCHEMA["required"]),
    )
    check(
        f"{label}: field types match schema",
        isinstance(obj.get("name"), str)
        and isinstance(obj.get("age"), int)
        and isinstance(obj.get("hobbies"), list),
    )


def main():
    _, api_key = load_api_keys()

    # Completions endpoint, object schema
    data = test_comp_request(api_key, BASE_URL, comp_request.copy(), n=1)
    validate_person("completion object", data["choices"][0]["text"])

    # Completions endpoint, n=2 (filters must be independent per generation)
    data = test_comp_request(api_key, BASE_URL, comp_request.copy(), n=2)
    for idx, choice in enumerate(data["choices"]):
        validate_person(f"completion object n=2 choice {idx}", choice["text"])

    # Completions endpoint, array schema (exercises the leading "[" constraint)
    data = test_comp_request(api_key, BASE_URL, comp_request_array.copy(), n=1)
    try:
        arr = json.loads(data["choices"][0]["text"])
        check("completion array: output is a JSON array", isinstance(arr, list))
        check("completion array: all items are strings", all(isinstance(x, str) for x in arr))
    except json.JSONDecodeError:
        check("completion array: output is valid JSON", False)

    # Chat completions endpoint
    data = test_chat_request(api_key, BASE_URL, chat_request.copy(), n=1)
    validate_person("chat object", data["choices"][0]["message"]["content"])

    print()
    if failures:
        print(f"{len(failures)} FAILED:")
        for f_ in failures:
            print(f"  - {f_}")
        exit(1)
    else:
        print("All JSON schema checks passed")


if __name__ == "__main__":
    main()
