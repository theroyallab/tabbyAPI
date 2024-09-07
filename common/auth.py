"""
This method of authorization is pretty insecure, but since TabbyAPI is a local
application, it should be fine.
"""

import secrets
import yaml
from fastapi import Header, HTTPException
from pydantic import BaseModel, Field
from loguru import logger
from typing import Union
from enum import Flag, auto
from abc import ABC, abstractmethod

from common.utils import unwrap

__all__ = ["ROLE", "auth"]


# RBAC roles
class ROLE(Flag):
    USER = auto()
    ADMIN = auto()


class API_KEY(BaseModel):
    """stores an API key"""

    key: str = Field(..., description="the API key value")
    role: ROLE = Field()


class AUTH_PROVIDER(ABC):
    @staticmethod
    @abstractmethod
    def add_api_key(role: ROLE) -> API_KEY:
        """add an API key"""

    @staticmethod
    @abstractmethod
    def set_api_key(role: ROLE, api_key: str) -> API_KEY:
        """add an existing API key"""

    @staticmethod
    @abstractmethod
    def remove_api_key(api_key: str) -> bool:
        """remove an API key"""

    @staticmethod
    @abstractmethod
    def check_api_key(api_key: str) -> Union[API_KEY, None]:
        """check if an API key is valid"""

    @staticmethod
    @abstractmethod
    def authenticate_api_key(api_key: str, role: ROLE) -> bool:
        """check if an api key has ROLE"""


class SIMPLE_AUTH_PROVIDER(AUTH_PROVIDER):
    api_keys: list[API_KEY] = []

    def __init__(self) -> None:
        try:
            with open("api_tokens.yml", "r", encoding="utf8") as auth_file:
                keys_dict: dict = unwrap(yaml.safe_load(auth_file), {})

                # load legacy keys
                admin_key = keys_dict.get("admin_key")
                if admin_key:
                    self.set_api_key(ROLE.ADMIN, admin_key)

                user_key = keys_dict.get("api_key")
                if user_key:
                    self.set_api_key(ROLE.USER, admin_key)

                # load new keys
                for role in ROLE :
                    role_keys = keys_dict.get(f"{role.name.lower()}_keys")
                    if role_keys:
                        for key in role_keys:
                            self.set_api_key(role, key)                    

        except FileNotFoundError:
            file = {}

            for role in ROLE :
                file[f"{role.name.lower()}_keys"] = [self.add_api_key(role).key for i in range(3)]

            print(file)
            with open("api_tokens.yml", "w", encoding="utf8") as auth_file:
                yaml.safe_dump(file, auth_file, default_flow_style=False)

        logger.info("API keys:")
        for key in self.api_keys:
            logger.info(f"{key.role.name} :\t {key.key}")
        logger.info(
            "If these keys get compromised, make sure to delete " +
                "api_tokens.yml and restart the server. Have fun!"
        )

    def add_api_key(self, role: ROLE) -> API_KEY:
        return self.set_api_key(api_key=secrets.token_hex(16), role=role)

    def set_api_key(self, role: ROLE, api_key: str) -> API_KEY:
        key = API_KEY(key=api_key, role=role)
        self.api_keys.append(key)
        return key

    def remove_api_key(self, api_key: str) -> bool:
        for key in self.api_keys:
            if key.key == api_key:
                self.api_keys.remove(key)
                return True
        return False

    def check_api_key(self, api_key: str) -> Union[API_KEY, None]:
        for key in self.api_keys:
            if key.key == api_key:
                return key
        return None

    def authenticate_api_key(self, api_key: str, role: ROLE) -> bool:
        key = self.check_api_key(api_key)
        if not key:
            return False
        return key.role & role  # if key.role in role


class NOAUTH_AUTH_PROVIDER(AUTH_PROVIDER):
    def add_api_key(self, role: ROLE) -> API_KEY:
        return API_KEY(key=secrets.token_hex(16), role=role)

    def set_api_key(self, role: ROLE, api_key: str) -> API_KEY:
        return API_KEY(key=secrets.token_hex(16), role=role)

    def remove_api_key(self, api_key: str) -> bool:
        return True

    def check_api_key(self, api_key: str) -> Union[API_KEY, None]:
        return API_KEY(key=secrets.token_hex(16), role=ROLE.ADMIN)

    def authenticate_api_key(self, api_key: str, role: ROLE) -> bool:
        return True


class AUTH_PROVIDER_CONTAINER:
    provider: AUTH_PROVIDER

    def load(self, disable_from_config: bool):
        """Load the authentication keys from api_tokens.yml. If the file does not
        exist, generate new keys and save them to api_tokens.yml."""

        # TODO: Make provider a paramater instead of disable_from_config
        provider = "noauth" if disable_from_config else "simple"

        # allows for more types of providers
        provider_class = {
            "noauth": NOAUTH_AUTH_PROVIDER,
            "simple": SIMPLE_AUTH_PROVIDER,
        }.get(provider)

        if not provider_class:
            raise Exception()

        if provider_class == NOAUTH_AUTH_PROVIDER:
            logger.warning(
                "Disabling authentication makes your instance vulnerable. "
                "Set the `disable_auth` flag to False in config.yml if you "
                "want to share this instance with others."
            )

        self.provider = provider_class()

    # by returning a dynamic dependency we can have one function
    # where we can specify what roles can access the endpoint
    def check_api_key(self, role: ROLE):
        """Check if the API key is valid."""

        async def check(
            x_api_key: str = Header(None), authorization: str = Header(None)
        ):
            if x_api_key:
                key = self.provider.authenticate_api_key(x_api_key, role)
                if not key:
                    raise HTTPException(401, "Invalid API key")
                return key

            if authorization:
                split_key = authorization.split(" ")
                if len(split_key) < 2:
                    raise HTTPException(401, "Invalid API key")
                key = self.provider.authenticate_api_key(split_key[1], role)
                if split_key[0].lower() != "bearer" or not key:
                    raise HTTPException(401, "Invalid API key")

                return key

            raise HTTPException(401, "Please provide an API key")

        return check


auth = AUTH_PROVIDER_CONTAINER()
