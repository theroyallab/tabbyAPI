"""
Functions for logging generation events.
"""
from typing import Dict
from pydantic import BaseModel


class LogConfig(BaseModel):
    """Logging preference config."""

    prompt: bool = False
    generation_params: bool = False


# Global reference to logging preferences
CONFIG = LogConfig()


def update_from_dict(options_dict: Dict[str, bool]):
    """Wrapper to set the logging config for generations"""
    global CONFIG  # pylint: disable=global-statement

    # Force bools on the dict
    for value in options_dict.values():
        if value is None:
            value = False

    CONFIG = LogConfig.model_validate(options_dict)


def broadcast_status():
    """Broadcasts the current logging status"""
    enabled = []
    if CONFIG.prompt:
        enabled.append("prompts")

    if CONFIG.generation_params:
        enabled.append("generation params")

    if len(enabled) > 0:
        print("Generation logging is enabled for: " + ", ".join(enabled))
    else:
        print("Generation logging is disabled")


def log_generation_params(**kwargs):
    """Logs generation parameters to console."""
    if CONFIG.generation_params:
        print(f"Generation options: {kwargs}\n")


def log_prompt(prompt: str):
    """Logs the prompt to console."""
    if CONFIG.prompt:
        print(f"Prompt: {prompt if prompt else 'Empty'}\n")


def log_response(response: str):
    """Logs the response to console."""
    if CONFIG.prompt:
        print(f"Response: {response if response else 'Empty'}\n")
