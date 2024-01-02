"""
Functions for logging generation events.
"""
from pydantic import BaseModel
from typing import Dict, Optional

from logger import init_logger

logger = init_logger(__name__)


class LogPreferences(BaseModel):
    """Logging preference config."""

    prompt: bool = False
    generation_params: bool = False


# Global reference to logging preferences
PREFERENCES = LogPreferences()


def update_from_dict(options_dict: Dict[str, bool]):
    """Wrapper to set the logging config for generations"""
    global PREFERENCES

    # Force bools on the dict
    for value in options_dict.values():
        if value is None:
            value = False

    PREFERENCES = LogPreferences.model_validate(options_dict)


def broadcast_status():
    """Broadcasts the current logging status"""
    enabled = []
    if PREFERENCES.prompt:
        enabled.append("prompts")

    if PREFERENCES.generation_params:
        enabled.append("generation params")

    if len(enabled) > 0:
        logger.info("Generation logging is enabled for: " + ", ".join(enabled))
    else:
        logger.info("Generation logging is disabled")


def log_generation_params(**kwargs):
    """Logs generation parameters to console."""
    if PREFERENCES.generation_params:
        logger.info(f"Generation options: {kwargs}\n")


def log_prompt(prompt: str, negative_prompt: Optional[str]):
    """Logs the prompt to console."""
    if PREFERENCES.prompt:
        formatted_prompt = "\n" + prompt
        logger.info(f"Prompt: {formatted_prompt if prompt else 'Empty'}\n")

        if negative_prompt:
            formatted_negative_prompt = "\n" + negative_prompt
            logger.info(f"Negative Prompt: {formatted_negative_prompt}\n")


def log_response(response: str):
    """Logs the response to console."""
    if PREFERENCES.prompt:
        formatted_response = "\n" + response
        logger.info(f"Response: {formatted_response if response else 'Empty'}\n")
