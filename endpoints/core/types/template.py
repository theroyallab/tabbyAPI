from pydantic import BaseModel, Field
from typing import List


class TemplateList(BaseModel):
    """Represents a list of templates."""

    object: str = "list"
    data: List[str] = Field(default_factory=list)


class TemplateSwitchRequest(BaseModel):
    """Request to switch a template."""

    name: str = Field(description="The name of the template to switch to")
