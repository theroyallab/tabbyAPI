from typing import Literal
from pydantic import BaseModel, Field

from common.health import UnhealthyEvent


class HealthCheckResponse(BaseModel):
    """System health status"""

    status: Literal["healthy", "unhealthy"] = Field(
        "healthy", description="System health status"
    )
    issues: list[UnhealthyEvent] = Field(
        default_factory=list, description="List of issues"
    )
