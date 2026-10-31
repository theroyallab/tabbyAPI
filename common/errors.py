from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


class ContextLengthExceededError(ValueError):
    """Raised when a tokenized prompt exceeds the loaded model's context limit."""


class EmptyPromptError(ValueError):
    """The prompt encodes to no tokens at all, leaving nothing to generate from."""


class GrammarParseError(ValueError):
    """Raised when a requested json_schema, regex, or grammar fails to compile.

    Constrained decoding refuses to fall back to unconstrained generation, so
    the request is rejected instead of silently served without the constraint.
    """


class LoopDetectedError(RuntimeError):
    """Raised when loop detection ends a generation whose request set loop_detect_error.

    The output so far is the model repeating itself, so the request fails with a
    server error the client can retry, instead of returning it as a response.
    """


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


class LoopDetectedHTTPException(HTTPException):
    """HTTP error for a generation that loop detection stopped, with loop_detect_error set."""

    def __init__(self, message: str):
        super().__init__(status_code=500, detail=message)


def loop_detected_error_content(message: str) -> dict:
    """
    Build an OpenAI-style server error for a detected loop. A 500 is what the
    OpenAI SDKs and most eval harnesses retry on their own.
    """

    return {
        "error": {
            "message": message,
            "type": "server_error",
            "param": None,
            "code": "loop_detected",
        }
    }


async def loop_detected_exception_handler(
    request: Request, exc: LoopDetectedHTTPException
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=loop_detected_error_content(exc.detail),
    )


def validate_context_requirements(
    context_len: int,
    max_seq_len: int,
    cache_capacity: int,
    draft_headroom: int = 0,
):
    """
    Reject a prompt the engine cannot start a job on. ExLlamaV3 caps a job's
    response at the room left in the cache past its prompt: one token per step
    plus the drafter's speculative window (draft_headroom = the generator's
    max_draft_tokens), which a verification pass writes beyond the budget. A
    request asking for more tokens than that is served up to the room and ends
    with finish_reason "length"; a prompt that leaves no room at all fails here.
    """

    if context_len > max_seq_len:
        raise ContextLengthExceededError(
            f"Prompt length {context_len} exceeds the available context size "
            f"of {max_seq_len} tokens (this model's maximum context length)"
        )

    # One output token, plus the one past it the engine budgets for (Job.prepare_for_queue)
    required_tokens = context_len + 2 + draft_headroom
    if required_tokens > cache_capacity:
        raise ContextLengthExceededError(
            f"Prompt length {context_len} leaves no room to generate: the job needs "
            f"{required_tokens} cache tokens (the prompt, one output token and the one past it"
            + (f", and a {draft_headroom}-token speculative window" if draft_headroom else "")
            + f"), which exceeds the available context size of {cache_capacity} tokens "
            "(this model's maximum context length)"
        )
