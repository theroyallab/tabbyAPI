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

def get_generator_error(exception: Exception):
    error_message = TabbyGeneratorErrorMessage(
        message = str(exception),
        trace = traceback.format_exc()
    )

    generator_error = TabbyGeneratorError(
        error = error_message
    )

    # Log and send the exception
    print(f"\n{generator_error.error.trace}")
    return generator_error.json()
