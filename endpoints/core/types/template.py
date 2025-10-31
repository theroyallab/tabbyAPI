from pydantic import BaseModel, Field


class TemplateList(BaseModel):
    """Represents a list of templates."""

    object: str = "list"
    data: list[str] = Field(default_factory=list)


class TemplateSwitchRequest(BaseModel):
    """Request to switch a template."""

    prompt_template_name: str
