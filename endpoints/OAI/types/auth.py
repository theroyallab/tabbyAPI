"""Types for auth requests."""

from pydantic import BaseModel


class AuthPermissionResponse(BaseModel):
    permission: str
