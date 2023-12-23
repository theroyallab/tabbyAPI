"""Common utilities for the tabbyAPI"""
import traceback
from typing import Optional

from pydantic import BaseModel

from logger import init_logger

logger = init_logger(__name__)


def load_progress(module, modules):
    """Wrapper callback for load progress."""
    yield module, modules


class TabbyGeneratorErrorMessage(BaseModel):
    """Common error types."""

    message: str
    trace: Optional[str] = None


class TabbyGeneratorError(BaseModel):
    """Common error types."""

    error: TabbyGeneratorErrorMessage


def get_generator_error(message: str):
    """Get a generator error."""
    error_message = TabbyGeneratorErrorMessage(
        message=message, trace=traceback.format_exc()
    )

    generator_error = TabbyGeneratorError(error=error_message)

    # Log and send the exception
    logger.error(generator_error.error.message)
    return get_sse_packet(generator_error.model_dump_json())


def get_sse_packet(json_data: str):
    """Get an SSE packet."""
    return f"data: {json_data}\n\n"


def unwrap(wrapped, default=None):
    """Unwrap function for Optionals."""
    if wrapped is None:
        return default

    return wrapped


def coalesce(*args):
    """Coalesce function for multiple unwraps."""
    return next((arg for arg in args if arg is not None), None)
