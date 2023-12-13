import secrets
import yaml
from fastapi import Header, HTTPException
from pydantic import BaseModel
from typing import Optional

"""
This method of authorization is pretty insecure, but since TabbyAPI is a local
application, it should be fine.
"""

class AuthKeys(BaseModel):
    api_key: str
    admin_key: str

    def verify_key(self, test_key: str, key_type: str):
        # Match statements are only available in python 3.10 and up
        if key_type == "admin_key":
            return test_key == self.admin_key
        elif key_type == "api_key":
            # Admin keys are valid for all API calls
            return test_key == self.api_key or test_key == self.admin_key
        else:
            return False

auth_keys: Optional[AuthKeys] = None

def load_auth_keys():
    global auth_keys
    try:
        with open("api_tokens.yml", "r", encoding = 'utf8') as auth_file:
            auth_keys_dict = yaml.safe_load(auth_file)
            auth_keys = AuthKeys.parse_obj(auth_keys_dict)
    except Exception as _:
        new_auth_keys = AuthKeys(
            api_key = secrets.token_hex(16),
            admin_key = secrets.token_hex(16)
        )
        auth_keys = new_auth_keys

        with open("api_tokens.yml", "w", encoding = "utf8") as auth_file:
            yaml.safe_dump(auth_keys.dict(), auth_file, default_flow_style=False)

    print(
        f"Your API key is: {auth_keys.api_key}\n"
        f"Your admin key is: {auth_keys.admin_key}\n\n"
        "If these keys get compromised, make sure to delete api_tokens.yml and restart the server. Have fun!"
    )

def check_api_key(x_api_key: str = Header(None), authorization: str = Header(None)):
    if x_api_key:
        if auth_keys.verify_key(x_api_key, "api_key"):
            return x_api_key
        else:
            raise HTTPException(401, "Invalid API key")
    elif authorization:
        split_key = authorization.split(" ")

        if len(split_key) < 2:
            raise HTTPException(401, "Invalid API key")
        elif split_key[0].lower() == "bearer" and auth_keys.verify_key(split_key[1], "api_key"):
            return authorization
        else:
            raise HTTPException(401, "Invalid API key")
    else:
        raise HTTPException(401, "Please provide an API key")

def check_admin_key(x_admin_key: str = Header(None), authorization: str = Header(None)):
    if x_admin_key:
        if auth_keys.verify_key(x_admin_key, "admin_key"):
            return x_admin_key
        else:
            raise HTTPException(401, "Invalid admin key")
    elif authorization:
        split_key = authorization.split(" ")

        if len(split_key) < 2:
            raise HTTPException(401, "Invalid admin key")
        elif split_key[0].lower() == "bearer" and auth_keys.verify_key(split_key[1], "admin_key"):
            return authorization
        else:
            raise HTTPException(401, "Invalid admin key")
    else:
        raise HTTPException(401, "Please provide an admin key")
