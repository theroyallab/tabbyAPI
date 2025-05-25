from typing import Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError


class ModelNotFoundError(Exception):
    """Raised when a requested model cannot be found."""

    def __init__(self, model_name: str):
        self.model_name = model_name


__all__ = [
    "ModelNotFoundError",
    "register_exception_handlers",
]


def _openai_error(
    message: str,
    type_: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
):
    return {"error": {"message": message, "type": type_, "param": param, "code": code}}


async def http_exception_handler(_request, exc: HTTPException):
    message = str(exc.detail)
    return JSONResponse(status_code=exc.status_code, content=_openai_error(message))


async def validation_exception_handler(_request, exc: RequestValidationError):
    if exc.errors():
        first = exc.errors()[0]
        message = first.get("msg", str(exc))
        loc = first.get("loc", [])
        param = ".".join(str(part) for part in loc) if loc else None
    else:
        message = str(exc)
        param = None
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=_openai_error(message, param=param),
    )


async def model_not_found_handler(_request, exc: ModelNotFoundError):
    message = f"The model '{exc.model_name}' does not exist."
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND, content=_openai_error(message)
    )


def register_exception_handlers(app: FastAPI):
    """Register OpenAI-style exception handlers on the given app."""
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ModelNotFoundError, model_not_found_handler)
