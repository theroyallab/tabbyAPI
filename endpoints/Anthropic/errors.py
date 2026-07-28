"""Error handling for the Anthropic API.

Anthropic clients parse a different error envelope than the one TabbyAPI
returns elsewhere, so responses on these routes are reshaped into:

    {"type": "error", "error": {"type": <error type>, "message": <message>}}

The reshaping lives in a route class rather than an app-level exception
handler so it also covers errors raised while solving dependencies (a failed
API key check) and request validation, neither of which reach the endpoint.
"""

from fastapi import HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from typing import Callable, Optional, Tuple

from common.networking import handle_request_error


# Anthropic error types by status code. Codes TabbyAPI raises that Anthropic
# does not define (422 from a template failure, 503 from an unloaded model)
# map onto the closest documented type.
ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "api_error",
    503: "api_error",
    529: "overloaded_error",
}


def error_type_for_status(status_code: int) -> str:
    """Get the Anthropic error type for a status code."""

    if status_code in ERROR_TYPES:
        return ERROR_TYPES[status_code]

    return "invalid_request_error" if status_code < 500 else "api_error"


def error_content(message: str, error_type: str) -> dict:
    """Build the error envelope Anthropic clients expect."""

    return {"type": "error", "error": {"type": error_type, "message": message}}


class AnthropicHTTPException(HTTPException):
    """An HTTP error carrying an explicit Anthropic error type."""

    def __init__(self, status_code: int, message: str, error_type: Optional[str] = None):
        super().__init__(status_code=status_code, detail=message)

        self.error_type = error_type or error_type_for_status(status_code)


def request_error(
    status_code: int, message: str, error_type: Optional[str] = None, exc_info: bool = False
) -> AnthropicHTTPException:
    """Log a request error and return the exception to raise."""

    error_message = handle_request_error(message, exc_info=exc_info).error.message

    return AnthropicHTTPException(status_code, error_message, error_type)


def exception_to_response(exc: Exception) -> Tuple[int, dict]:
    """Map an exception raised while serving a route onto an error response."""

    if isinstance(exc, AnthropicHTTPException):
        return exc.status_code, error_content(exc.detail, exc.error_type)

    if isinstance(exc, HTTPException):
        # Raised by shared TabbyAPI code: auth, inline model loading, chat
        # template rendering, context length
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return exc.status_code, error_content(detail, error_type_for_status(exc.status_code))

    if isinstance(exc, RequestValidationError):
        return 422, error_content(str(exc.errors()), "invalid_request_error")

    raise exc


class AnthropicRoute(APIRoute):
    """Route class that returns errors in the Anthropic envelope."""

    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def anthropic_route_handler(request: Request) -> Response:
            try:
                return await original_route_handler(request)
            except (HTTPException, RequestValidationError) as exc:
                status_code, content = exception_to_response(exc)

                return JSONResponse(status_code=status_code, content=content)

        return anthropic_route_handler
