import asyncio
from typing import Union
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from collections import deque
from functools import partial


class UnhealthyEvent(BaseModel):
    """Represents an error that makes the system unhealthy"""

    time: datetime = Field(
        default_factory=partial(datetime.now, timezone.utc),
        description="Time the error occurred in UTC time",
    )
    description: str = Field("Unknown error", description="The error message")


class HealthManagerClass:
    """Class to manage the health global state"""

    def __init__(self):
        # limit the max stored errors to 100 to avoid a memory leak
        self.issues: deque[UnhealthyEvent] = deque(maxlen=100)
        self._lock = asyncio.Lock()

    async def add_unhealthy_event(self, error: Union[str, Exception]):
        """Add a new unhealthy event"""
        async with self._lock:
            if isinstance(error, Exception):
                error = f"{error.__class__.__name__}: {str(error)}"
            self.issues.append(UnhealthyEvent(description=error))

    async def is_service_healthy(self) -> tuple[bool, list[UnhealthyEvent]]:
        """Check if the service is healthy"""
        async with self._lock:
            healthy = len(self.issues) == 0
            return healthy, list(self.issues)


# Create an instance of the global state manager
HealthManager = HealthManagerClass()
