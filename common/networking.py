"""Common utility functions"""

import asyncio
import itertools
import json
import platform
import socket
import traceback
from fastapi import Depends, HTTPException, Request
from loguru import logger
from common.logger import xlogger
from pydantic import BaseModel
from typing import Optional
from uuid import uuid4

from common.errors import context_length_error_content
from common.tabby_config import config


def get_sse_ping_interval() -> int:
    """SSE keep-alive ping interval in seconds, or effectively never if disabled."""

    from sys import maxsize

    interval = config.network.sse_ping_interval
    return interval if interval else maxsize


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


def get_context_length_generator_error(message: str):
    """Get an OpenAI-compatible context overflow error for an active stream."""

    handle_request_error(message, exc_info=False)
    return json.dumps(context_length_error_content(message))


def handle_request_error(message: str, exc_info: bool = True):
    """Log a request error to the console."""

    trace = traceback.format_exc()
    send_trace = config.network.send_tracebacks

    error_message = TabbyRequestErrorMessage(message=message, trace=trace if send_trace else None)

    request_error = TabbyRequestError(error=error_message)

    # Log the error and provided message to the console
    if trace and exc_info:
        xlogger.error("Error", {"trace": trace, "message": message}, details=trace)

    logger.error(f"Sent to request: {message}")

    return request_error


def handle_request_disconnect(message: str):
    """Wrapper for handling for request disconnection."""

    xlogger.error(message)


class DisconnectHandler:
    """
    Tracks whether the client of a request has gone away.

    A background task owns the request's ASGI receive channel and flags the
    handler as soon as an ``http.disconnect`` message arrives, so ``poll()``
    is a plain flag check that never suspends the caller. This keeps the
    per-token consumer loop cheap and lets it run in lockstep with the
    generator instead of yielding to it while the request status is queried.
    """

    def __init__(
        self,
        request: Request,
        description: str,
    ):
        self.request = request
        self.abort_event = asyncio.Event()
        self.disconnected = False
        self.cleanup_tasks = {}
        self.description = description

        self._reported = False
        self._watcher = asyncio.create_task(self._watch())

    async def _watch(self):
        """Wait for the client to disconnect and flag it."""

        try:
            # The request body has already been consumed at this point, so the
            # only messages left on the channel are stray http.request events
            # (which are ignored) and the eventual http.disconnect. Uvicorn also
            # sends http.disconnect once the response completes, so this task
            # always terminates on its own even if cleanup() is never called.
            while True:
                message = await self.request.receive()
                if message["type"] == "http.disconnect":
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"Disconnect watcher for {self.description} stopped: {exc}")
            return

        self.disconnected = True
        self.abort_event.set()

    async def poll(self):
        """
        Check whether the request has disconnected. This does not suspend the caller. Once the
        request is disconnected, runs scheduled cleanup tasks and raises asyncio.CancelledError.
        Caller is responsible for forwarding the error back to the endpoint function. The endpoint
        fn should call poll() at least once before returning a non-canceled response.
        """

        if not self.disconnected:
            return

        # Trigger any cleanup tasks
        await self.cleanup()

        # Log and raise
        if not self._reported:
            xlogger.warning(f"{self.description}: client disconnected, generation cancelled")
            self._reported = True

        raise asyncio.CancelledError(f"{self.description}: client disconnected")

    async def add_cleanup_task(self, key, func, args):
        # Intentionally strict
        assert key not in self.cleanup_tasks
        self.cleanup_tasks[key] = (func, args)

    async def finish(self, key):
        # Intentionally strict
        del self.cleanup_tasks[key]

    # Safe to call redundantly, each cleanup task must be called exactly once
    async def cleanup(self):
        self._watcher.cancel()
        for func, args in self.cleanup_tasks.values():
            await func(*args)
        self.cleanup_tasks = {}


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


def port_bind_error(host: str, port: int) -> Optional[str]:
    """
    Why the server could not listen on host:port, or None if it can.

    Checks by binding a throwaway socket to the same address the server will
    bind, with the same options, so the answer matches what the server is about
    to find out. (Connecting to localhost instead, as this used to, answers a
    different question: it reports a listener on the loopback interface whether
    or not it blocks our bind, e.g. a Docker port mapping while the server is
    configured for another interface, and misses listeners on other interfaces.)
    """

    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
    except socket.gaierror as ex:
        return f"{host} is not a usable address: {ex}"

    for family, socktype, proto, _, sockaddr in addresses:
        try:
            probe = socket.socket(family, socktype, proto)
        except OSError:
            # Address family not supported here (e.g. IPv6 disabled)
            continue

        with probe:
            # The server's event loop sets the same options: reuse of a port
            # left in TIME_WAIT on POSIX (never on Windows, where the flag
            # lets a second listener take over the port), one socket per
            # address family
            if family == socket.AF_INET6 and hasattr(socket, "IPV6_V6ONLY"):
                probe.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            if platform.system() != "Windows":
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            try:
                probe.bind(sockaddr)
            except OSError as ex:
                reason = ex.strerror or str(ex)
                return f"{reason} ({sockaddr[0]}:{sockaddr[1]})"

    return None


# Short per-process serial for console log lines; the UUID stays the API-facing id
_request_serials = itertools.count(1)


async def add_request_id(request: Request):
    """FastAPI depends to add a UUID and a console serial to a request's state."""

    request.state.id = uuid4().hex
    request.state.serial = next(_request_serials)
    return request


def request_tag(request: Request) -> str:
    """Short tag identifying a request in console logs, e.g. "#12"."""

    serial = getattr(request.state, "serial", None)
    return f"#{serial}" if serial is not None else f"#{request.state.id[:8]}"


async def log_request(request: Request):
    """FastAPI depends to log a request to the user."""

    log_message = [f"{request_tag(request)} {request.method} request (ID {request.state.id}):"]

    log_message.append(f"URL: {request.url}")
    log_message.append(f"Headers: {dict(request.headers)}")

    if request.method != "GET":
        body_bytes = await request.body()
        if body_bytes:
            body = json.loads(body_bytes.decode("utf-8"))

            log_message.append(f"Body: {dict(body)}")

    xlogger.info("Request", dict(request), details="\n".join(log_message))


def get_global_depends():
    """Returns global dependencies for a FastAPI app."""

    depends = [Depends(add_request_id)]

    if config.logging.log_requests:
        depends.append(Depends(log_request))

    return depends
