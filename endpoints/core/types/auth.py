"""Types for auth requests."""

from pydantic import BaseModel, Field


class AuthPermissionResponse(BaseModel):
    permission: str = Field(description="The permission level of the API key")
