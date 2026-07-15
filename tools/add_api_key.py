"""
Generate a random 128-bit API key, add it to api_tokens.yml and print it.

The running server picks the new key up automatically; no restart needed.
Run from the TabbyAPI directory: python tools/add_api_key.py
"""

import os
import secrets
import sys
import tempfile

from ruamel.yaml import YAML

AUTH_FILE = "api_tokens.yml"


def main():
    yaml = YAML()  # round-trip mode preserves comments and formatting

    try:
        with open(AUTH_FILE, "r", encoding="utf8") as auth_file:
            auth_keys = yaml.load(auth_file)
    except FileNotFoundError:
        print(
            f"{AUTH_FILE} not found. Start the server once to generate it, "
            "or run this script from the TabbyAPI directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    new_key = secrets.token_hex(16)

    api_key = auth_keys.get("api_key")
    if api_key is None:
        auth_keys["api_key"] = [new_key]
    elif isinstance(api_key, str):
        auth_keys["api_key"] = [api_key, new_key]
    else:
        api_key.append(new_key)

    # Write to a temporary file and rename so the server's file watcher
    # never sees a partial write
    fd, temp_path = tempfile.mkstemp(dir=".", prefix=".api_tokens_", suffix=".yml")
    try:
        with os.fdopen(fd, "w", encoding="utf8") as temp_file:
            yaml.dump(auth_keys, temp_file)
        os.replace(temp_path, AUTH_FILE)
    except BaseException:
        os.unlink(temp_path)
        raise

    print(f"Added a new API key to {AUTH_FILE}:", file=sys.stderr)
    print(new_key)


if __name__ == "__main__":
    main()
