import traceback
from pydantic import BaseModel
from typing import Optional

# Wrapper callback for load progress
def load_progress(module, modules):
    yield module, modules

# Common error types
class TabbyGeneratorErrorMessage(BaseModel):
    message: str
    trace: Optional[str] = None

class TabbyGeneratorError(BaseModel):
    error: TabbyGeneratorErrorMessage

def get_generator_error(message: str):
    error_message = TabbyGeneratorErrorMessage(
        message = message,
        trace = traceback.format_exc()
    )

    generator_error = TabbyGeneratorError(
        error = error_message
    )

    # Log and send the exception
    print(f"\n{generator_error.error.trace}")
    return get_sse_packet(generator_error.model_dump_json())

def get_sse_packet(json_data: str):
    return f"data: {json_data}\n\n"

# Unwrap function for Optionals
def unwrap(wrapped, default = None):
    if wrapped is None:
        return default
    else:
        return wrapped

# Coalesce function for multiple unwraps
def coalesce(*args):
    return next((arg for arg in args if arg is not None), None)
