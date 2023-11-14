import secrets
import yaml
from fastapi import Header, HTTPException
from typing import Optional

"""
This method of authorization is pretty insecure, but since TabbyAPI is a local
application, it should be fine.
"""

class AuthKeys:
    api_key: str
    admin_key: str

    def __init__(self, api_key: str, admin_key: str):
        self.api_key = api_key
        self.admin_key = admin_key

auth_keys: Optional[AuthKeys] = None

def load_auth_keys():
    global auth_keys
    try:
        with open("api_tokens.yml", "r") as auth_file:
            auth_keys = yaml.safe_load(auth_file)
    except:
        new_auth_keys = AuthKeys(
            api_key = secrets.token_hex(16),
            admin_key = secrets.token_hex(16)
        )
        auth_keys = new_auth_keys

        with open("api_tokens.yml", "w") as auth_file:
            yaml.dump(auth_keys, auth_file)

def check_api_key(x_api_key: str = Header(None), authorization: str = Header(None)):
    if x_api_key and x_api_key == auth_keys.api_key:
        return x_api_key
    elif authorization:
        split_key = authorization.split(" ")
        if split_key[0].lower() == "bearer" and split_key[1] == auth_keys.api_key:
            return authorization
    else:
        raise HTTPException(401, "Invalid API key")

def check_admin_key(x_admin_key: str = Header(None), authorization: str = Header(None)):
    if x_admin_key and x_admin_key == auth_keys.admin_key:
        return x_admin_key
    elif authorization:
        split_key = authorization.split(" ")
        if split_key[0].lower() == "bearer" and split_key[1] == auth_keys.admin_key:
            return authorization
    else:
        raise HTTPException(401, "Invalid admin key")
