"""
This method of authorization is pretty insecure, but since TabbyAPI is a local
application, it should be fine.
"""
import secrets
import yaml
from fastapi import Header, HTTPException
from pydantic import BaseModel
from typing import Optional

from common.logger import init_logger

logger = init_logger(__name__)


class AuthKeys(BaseModel):
    """
    This class represents the authentication keys for the application.
    It contains two types of keys: 'api_key' and 'admin_key'.
    The 'api_key' is used for general API calls, while the 'admin_key'
    is used for administrative tasks. The class also provides a method
    to verify if a given key matches the stored 'api_key' or 'admin_key'.
    """

    api_key: str
    admin_key: str

    def verify_key(self, test_key: str, key_type: str):
        """Verify if a given key matches the stored key."""
        if key_type == "admin_key":
            return test_key == self.admin_key
        if key_type == "api_key":
            # Admin keys are valid for all API calls
            return test_key == self.api_key or test_key == self.admin_key
        return False


AUTH_KEYS: Optional[AuthKeys] = None
DISABLE_AUTH: bool = False


def load_auth_keys(disable_from_config: bool):
    """Load the authentication keys from api_tokens.yml. If the file does not
    exist, generate new keys and save them to api_tokens.yml."""
    global AUTH_KEYS
    global DISABLE_AUTH

    DISABLE_AUTH = disable_from_config
    if disable_from_config:
        logger.warning(
            "Disabling authentication makes your instance vulnerable. "
            "Set the `disable_auth` flag to False in config.yml if you "
            "want to share this instance with others."
        )

        return

    try:
        with open("api_tokens.yml", "r", encoding="utf8") as auth_file:
            auth_keys_dict = yaml.safe_load(auth_file)
            AUTH_KEYS = AuthKeys.model_validate(auth_keys_dict)
    except FileNotFoundError:
        new_auth_keys = AuthKeys(
            api_key=secrets.token_hex(16), admin_key=secrets.token_hex(16)
        )
        AUTH_KEYS = new_auth_keys

        with open("api_tokens.yml", "w", encoding="utf8") as auth_file:
            yaml.safe_dump(AUTH_KEYS.model_dump(), auth_file, default_flow_style=False)

    logger.info(
        f"Your API key is: {AUTH_KEYS.api_key}\n"
        f"Your admin key is: {AUTH_KEYS.admin_key}\n\n"
        "If these keys get compromised, make sure to delete api_tokens.yml "
        "and restart the server. Have fun!"
    )


def check_api_key(x_api_key: str = Header(None), authorization: str = Header(None)):
    """Check if the API key is valid."""

    # Allow request if auth is disabled
    if DISABLE_AUTH:
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


def check_admin_key(x_admin_key: str = Header(None), authorization: str = Header(None)):
    """Check if the admin key is valid."""

    # Allow request if auth is disabled
    if DISABLE_AUTH:
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
