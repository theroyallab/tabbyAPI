"""
This method of authorization is pretty insecure, but since TabbyAPI is a local
application, it should be fine.
"""

import asyncio
import io
import os
import secrets
from typing import List, Optional, Union

import aiofiles
from fastapi import Header, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, PrivateAttr
from ruamel.yaml import YAML

from common.logger import xlogger
from common.utils import coalesce

AUTH_FILE = "api_tokens.yml"

# Seconds between checks for changes to the auth file
AUTH_FILE_POLL_INTERVAL = 2.0


class AuthKeys(BaseModel):
    """
    This class represents the authentication keys for the application.
    It contains two types of keys: 'api_key' and 'admin_key'.
    The 'api_key' is used for general API calls, while the 'admin_key'
    is used for administrative tasks. The class also provides a method
    to verify if a given key matches the stored 'api_key' or 'admin_key'.

    api_key accepts either a single key or a list of keys, so access can
    be granted (and revoked) per user. There is always exactly one admin_key.
    """

    api_key: Union[str, List[str]]
    admin_key: str

    _api_key_set: set = PrivateAttr(default_factory=set)

    def model_post_init(self, __context):
        if isinstance(self.api_key, str):
            self._api_key_set = {self.api_key}
        else:
            self._api_key_set = set(self.api_key)

    def verify_key(self, test_key: str, key_type: str):
        """Verify if a given key matches the stored key."""
        if key_type == "admin_key":
            return test_key == self.admin_key
        if key_type == "api_key":
            # Admin keys are valid for all API calls
            return test_key in self._api_key_set or test_key == self.admin_key
        return False


# Global auth constants
AUTH_KEYS: Optional[AuthKeys] = None
DISABLE_AUTH: bool = False

# Serializes reloads of the auth file. Reads don't need the lock: the working
# set of keys is swapped in as one immutable object, and every check function
# takes its own reference.
_reload_lock = asyncio.Lock()
_watch_task: Optional[asyncio.Task] = None


async def _read_auth_file(path: str = AUTH_FILE) -> AuthKeys:
    """Read and validate the auth keys file."""

    yaml = YAML(typ=["rt", "safe"])

    async with aiofiles.open(path, "r", encoding="utf8") as auth_file:
        contents = await auth_file.read()
        auth_keys_dict = yaml.load(contents)
        return AuthKeys.model_validate(auth_keys_dict)


async def _watch_auth_file():
    """
    Poll the auth file for changes and reload the working set of keys.

    A failed reload (partial write, invalid YAML, missing keys) keeps the
    previous keys. Ongoing requests are unaffected either way; a reload only
    changes which keys validate for future requests.
    """

    global AUTH_KEYS

    try:
        last_mtime = os.stat(AUTH_FILE).st_mtime
    except OSError:
        last_mtime = None

    while True:
        await asyncio.sleep(AUTH_FILE_POLL_INTERVAL)

        try:
            mtime = os.stat(AUTH_FILE).st_mtime
        except OSError:
            continue

        if mtime == last_mtime:
            continue
        last_mtime = mtime

        async with _reload_lock:
            try:
                AUTH_KEYS = await _read_auth_file()
            except Exception as exc:
                xlogger.warning(f"Failed to reload {AUTH_FILE}, keeping the previous keys: {exc}")
                continue

        xlogger.info(
            f"Reloaded auth keys from {AUTH_FILE} ({len(AUTH_KEYS._api_key_set)} API key(s))."
        )


def _format_api_keys(auth_keys: AuthKeys) -> str:
    if isinstance(auth_keys.api_key, str):
        return auth_keys.api_key
    return ", ".join(auth_keys.api_key)


async def load_auth_keys(disable_from_config: bool):
    """Load the authentication keys from api_tokens.yml. If the file does not
    exist, generate new keys and save them to api_tokens.yml."""
    global AUTH_KEYS
    global DISABLE_AUTH
    global _watch_task

    DISABLE_AUTH = disable_from_config
    if disable_from_config:
        xlogger.warning(
            "Disabling authentication makes your instance vulnerable. "
            "Set the `disable_auth` flag to False in config.yml if you "
            "want to share this instance with others."
        )

        return

    try:
        AUTH_KEYS = await _read_auth_file()
    except FileNotFoundError:
        new_auth_keys = AuthKeys(api_key=secrets.token_hex(16), admin_key=secrets.token_hex(16))
        AUTH_KEYS = new_auth_keys

        yaml = YAML(typ=["rt", "safe"])
        async with aiofiles.open(AUTH_FILE, "w", encoding="utf8") as auth_file:
            string_stream = io.StringIO()
            yaml.dump(AUTH_KEYS.model_dump(), string_stream)

            await auth_file.write(string_stream.getvalue())

    # Reload the keys whenever the file changes, so keys can be added or
    # revoked without a server restart
    if _watch_task is None:
        _watch_task = asyncio.create_task(_watch_auth_file())

    logger.info(
        f"Your API key is: {_format_api_keys(AUTH_KEYS)}\n"
        f"Your admin key is: {AUTH_KEYS.admin_key}\n"
        "If these keys get compromised, make sure to delete api_tokens.yml "
        "and restart the server. Have fun!"
    )


def get_key_permission(request: Request):
    """
    Gets the key permission from a request.

    Internal only! Use the depends functions for incoming requests.
    """

    # Give full admin permissions if auth is disabled
    if DISABLE_AUTH:
        return "admin"

    auth_keys = AUTH_KEYS

    # Hyphens are okay here
    test_key = coalesce(
        request.headers.get("x-admin-key"),
        request.headers.get("x-api-key"),
        request.headers.get("authorization"),
    )

    if test_key is None:
        raise ValueError("The provided authentication key is missing.")

    if test_key.lower().startswith("bearer"):
        test_key = test_key.split(" ")[1]

    if auth_keys.verify_key(test_key, "admin_key"):
        return "admin"
    elif auth_keys.verify_key(test_key, "api_key"):
        return "api"
    else:
        raise ValueError("The provided authentication key is invalid.")


async def check_api_key(x_api_key: str = Header(None), authorization: str = Header(None)):
    """Check if the API key is valid."""

    # Allow request if auth is disabled
    if DISABLE_AUTH:
        return

    auth_keys = AUTH_KEYS

    if x_api_key:
        if not auth_keys.verify_key(x_api_key, "api_key"):
            raise HTTPException(401, "Invalid API key")
        return x_api_key

    if authorization:
        split_key = authorization.split(" ")
        if len(split_key) < 2:
            raise HTTPException(401, "Invalid API key")
        if split_key[0].lower() != "bearer" or not auth_keys.verify_key(split_key[1], "api_key"):
            raise HTTPException(401, "Invalid API key")

        return authorization

    raise HTTPException(401, "Please provide an API key")


async def check_admin_key(x_admin_key: str = Header(None), authorization: str = Header(None)):
    """Check if the admin key is valid."""

    # Allow request if auth is disabled
    if DISABLE_AUTH:
        return

    auth_keys = AUTH_KEYS

    if x_admin_key:
        if not auth_keys.verify_key(x_admin_key, "admin_key"):
            raise HTTPException(401, "Invalid admin key")
        return x_admin_key

    if authorization:
        split_key = authorization.split(" ")
        if len(split_key) < 2:
            raise HTTPException(401, "Invalid admin key")
        if split_key[0].lower() != "bearer" or not auth_keys.verify_key(split_key[1], "admin_key"):
            raise HTTPException(401, "Invalid admin key")
        return authorization

    raise HTTPException(401, "Please provide an admin key")
