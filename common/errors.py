from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


class ContextLengthExceededError(ValueError):
    """Raised when a tokenized prompt exceeds the loaded model's context limit."""


class ContextLengthHTTPException(HTTPException):
    """HTTP error for OpenAI-compatible context overflow responses."""

    def __init__(self, message: str):
        super().__init__(status_code=400, detail=message)


def context_length_error_content(message: str) -> dict:
    """Build an OpenAI-compatible context overflow error."""

    return {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": None,
            "code": "context_length_exceeded",
        }
    }


async def context_length_exception_handler(
    request: Request, exc: ContextLengthHTTPException
) -> JSONResponse:
    """Return the OpenAI error shape expected by compatible clients."""

    return JSONResponse(
        status_code=exc.status_code,
        content=context_length_error_content(exc.detail),
    )


def validate_context_requirements(
    context_len: int,
    max_seq_len: int,
    max_tokens: int,
    cache_capacity: int,
    max_rq_tokens: int | None = None,
    allocation_boundary: int = 256,
):
    """Validate the initial cache allocation required by an ExLlamaV3 job."""

    if context_len > max_seq_len:
        raise ContextLengthExceededError(
            f"Prompt length {context_len} exceeds the available context size "
            f"of {max_seq_len} tokens"
        )

    if max_tokens <= 0:
        max_tokens = max_seq_len - context_len - 1

    if max_rq_tokens is not None:
        required_tokens = (
            (context_len - 1 + max_rq_tokens + allocation_boundary - 1) // allocation_boundary
        ) * allocation_boundary
    else:
        required_tokens = context_len + max_tokens

    if required_tokens > cache_capacity:
        raise ContextLengthExceededError(
            f"Initial job allocation requires {required_tokens} cache tokens, "
            f"which exceeds the available context size of {cache_capacity} tokens"
        )
