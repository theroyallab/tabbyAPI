"""
This method of authorization is pretty insecure, but since TabbyAPI is a local
application, it should be fine.
"""

from functools import partial
import aiofiles
import io
import secrets
from ruamel.yaml import YAML
from fastapi import Header, HTTPException, Request
from pydantic import BaseModel, Field, SecretStr
from loguru import logger
from typing import Optional

from common.utils import coalesce
from common.tabby_config import config


class AuthKeys(BaseModel):
    """
    This class represents the authentication keys for the application.
    It contains two types of keys: 'api_key' and 'admin_key'.
    The 'api_key' is used for general API calls, while the 'admin_key'
    is used for administrative tasks. The class also provides a method
    to verify if a given key matches the stored 'api_key' or 'admin_key'.
    """

    api_key: SecretStr = Field(default_factory=partial(secrets.token_hex, 16))
    admin_key: SecretStr = Field(default_factory=partial(secrets.token_hex, 16))

    def verify_key(self, test_key: str, key_type: str):
        """Verify if a given key matches the stored key."""
        if key_type == "admin_key":
            return test_key == self.admin_key
        if key_type == "api_key":
            # Admin keys are valid for all API calls
            return test_key == self.api_key or test_key == self.admin_key
        return False


# Global auth constants
AUTH_KEYS: Optional[AuthKeys] = None


async def load_auth_keys():
    """Load the authentication keys from api_tokens.yml. If the file does not
    exist, generate new keys and save them to api_tokens.yml."""
    global AUTH_KEYS

    if config.network.disable_auth:
        logger.warning(
            "Disabling authentication makes your instance vulnerable. "
            "Set the `disable_auth` flag to False in config.yml if you "
            "want to share this instance with others."
        )

        return

    # Create a temporary YAML parser
    yaml = YAML(typ=["rt", "safe"])

    try:
        async with aiofiles.open("api_tokens.yml", "r", encoding="utf8") as auth_file:
            contents = await auth_file.read()
            auth_keys_dict = yaml.load(contents)
            AUTH_KEYS = AuthKeys.model_validate(auth_keys_dict)
    except FileNotFoundError:
        new_auth_keys = AuthKeys()
        AUTH_KEYS = new_auth_keys

        async with aiofiles.open("api_tokens.yml", "w", encoding="utf8") as auth_file:
            string_stream = io.StringIO()
            yaml.dump(AUTH_KEYS.model_dump(), string_stream)

            await auth_file.write(string_stream.getvalue())

    logger.info(
        f"Your API key is: {AUTH_KEYS.api_key.get_secret_value()}\n"
        f"Your admin key is: {AUTH_KEYS.admin_key.get_secret_value()}\n\n"
        "If these keys get compromised, make sure to delete api_tokens.yml "
        "and restart the server. Have fun!"
    )


def get_key_permission(request: Request):
    """
    Gets the key permission from a request.

    Internal only! Use the depends functions for incoming requests.
    """

    # Give full admin permissions if auth is disabled
    if config.network.disable_auth:
        return "admin"

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

    if AUTH_KEYS.verify_key(test_key, "admin_key"):
        return "admin"
    elif AUTH_KEYS.verify_key(test_key, "api_key"):
        return "api"
    else:
        raise ValueError("The provided authentication key is invalid.")


async def check_api_key(
    x_api_key: str = Header(None), authorization: str = Header(None)
):
    """Check if the API key is valid."""

    # Allow request if auth is disabled
    if config.network.disable_auth:
        return

    if x_api_key:
        if not AUTH_KEYS.verify_key(x_api_key, "api_key"):
            raise HTTPException(401, "Invalid API key")
        return x_api_key

    if authorization:
        split_key = authorization.split(" ")
        if len(split_key) < 2:
            raise HTTPException(401, "Invalid API key")
        if split_key[0].lower() != "bearer" or not AUTH_KEYS.verify_key(
            split_key[1], "api_key"
        ):
            raise HTTPException(401, "Invalid API key")

        return authorization

    raise HTTPException(401, "Please provide an API key")


async def check_admin_key(
    x_admin_key: str = Header(None), authorization: str = Header(None)
):
    """Check if the admin key is valid."""

    # Allow request if auth is disabled
    if config.network.disable_auth:
        return

    if x_admin_key:
        if not AUTH_KEYS.verify_key(x_admin_key, "admin_key"):
            raise HTTPException(401, "Invalid admin key")
        return x_admin_key

    if authorization:
        split_key = authorization.split(" ")
        if len(split_key) < 2:
            raise HTTPException(401, "Invalid admin key")
        if split_key[0].lower() != "bearer" or not AUTH_KEYS.verify_key(
            split_key[1], "admin_key"
        ):
            raise HTTPException(401, "Invalid admin key")
        return authorization

    raise HTTPException(401, "Please provide an admin key")
