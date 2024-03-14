"""Common utility functions"""

import socket
import traceback
from loguru import logger
from pydantic import BaseModel
from typing import Optional

from common.concurrency import release_semaphore


def load_progress(module, modules):
    """Wrapper callback for load progress."""
    yield module, modules


class TabbyRequestErrorMessage(BaseModel):
    """Common request error type."""

    message: str
    trace: Optional[str] = None


class TabbyRequestError(BaseModel):
    """Common request error type."""

    error: TabbyRequestErrorMessage


def get_generator_error(message: str, exc_info: bool = True):
    """Get a generator error."""

    generator_error = handle_request_error(message)

    return generator_error.model_dump_json()


def handle_request_error(message: str, exc_info: bool = True):
    """Log a request error to the console."""

    error_message = TabbyRequestErrorMessage(
        message=message, trace=traceback.format_exc()
    )

    request_error = TabbyRequestError(error=error_message)

    # Log the error and provided message to the console
    if error_message.trace and exc_info:
        logger.error(error_message.trace)

    logger.error(f"Sent to request: {message}")

    return request_error


def handle_request_disconnect(message: str):
    """Wrapper for handling for request disconnection."""

    release_semaphore()
    logger.error(message)


def unwrap(wrapped, default=None):
    """Unwrap function for Optionals."""
    if wrapped is None:
        return default

    return wrapped


def coalesce(*args):
    """Coalesce function for multiple unwraps."""
    return next((arg for arg in args if arg is not None), None)


def prune_dict(input_dict):
    """Trim out instances of None from a dictionary"""

    return {k: v for k, v in input_dict.items() if v is not None}


def is_port_in_use(port: int) -> bool:
    """
    Checks if a port is in use

    From https://stackoverflow.com/questions/2470971/fast-way-to-test-if-a-port-is-in-use-using-python
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0
