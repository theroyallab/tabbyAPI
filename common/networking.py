"""Common utility functions"""

import asyncio
import socket
import traceback
from fastapi import HTTPException, Request
from loguru import logger
from pydantic import BaseModel
from typing import Optional
from uuid import uuid4

from common import config
from common.utils import unwrap


class TabbyRequestErrorMessage(BaseModel):
    """Common request error type."""

    message: str
    trace: Optional[str] = None


class TabbyRequestError(BaseModel):
    """Common request error type."""

    error: TabbyRequestErrorMessage


def get_generator_error(message: str, exc_info: bool = True):
    """Get a generator error."""

    generator_error = handle_request_error(message, exc_info)

    return generator_error.model_dump_json()


def handle_request_error(message: str, exc_info: bool = True):
    """Log a request error to the console."""

    trace = traceback.format_exc()
    send_trace = unwrap(config.network_config().get("send_tracebacks"), False)

    error_message = TabbyRequestErrorMessage(
        message=message, trace=trace if send_trace else None
    )

    request_error = TabbyRequestError(error=error_message)

    # Log the error and provided message to the console
    if trace and exc_info:
        logger.error(trace)

    logger.error(f"Sent to request: {message}")

    return request_error


def handle_request_disconnect(message: str):
    """Wrapper for handling for request disconnection."""

    logger.error(message)


async def request_disconnect_loop(request: Request):
    """Polls for a starlette request disconnect."""

    while not await request.is_disconnected():
        await asyncio.sleep(0.5)


async def run_with_request_disconnect(
    request: Request, call_task: asyncio.Task, disconnect_message: str
):
    """Utility function to cancel if a request is disconnected."""

    _, unfinished = await asyncio.wait(
        [
            call_task,
            asyncio.create_task(request_disconnect_loop(request)),
        ],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in unfinished:
        task.cancel()

    try:
        return call_task.result()
    except (asyncio.CancelledError, asyncio.InvalidStateError) as ex:
        handle_request_disconnect(disconnect_message)
        raise HTTPException(422, disconnect_message) from ex


def is_port_in_use(port: int) -> bool:
    """
    Checks if a port is in use

    From https://stackoverflow.com/questions/2470971/fast-way-to-test-if-a-port-is-in-use-using-python
    """

    test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    test_socket.settimeout(1)
    with test_socket:
        return test_socket.connect_ex(("localhost", port)) == 0


async def add_request_id(request: Request):
    """FastAPI depends to add a UUID to a request's state."""

    request.state.id = uuid4().hex
    return request
