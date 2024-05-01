"""
Functions for logging generation events.
"""

from pydantic import BaseModel
from loguru import logger
from typing import Dict, Optional


class GenLogPreferences(BaseModel):
    """Logging preference config."""

    prompt: bool = False
    generation_params: bool = False


# Global logging preferences constant
PREFERENCES = GenLogPreferences()


def update_from_dict(options_dict: Dict[str, bool]):
    """Wrapper to set the logging config for generations"""
    global PREFERENCES

    # Force bools on the dict
    for value in options_dict.values():
        if value is None:
            value = False

    PREFERENCES = GenLogPreferences.model_validate(options_dict)


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


def log_metrics(
    generated_tokens: int,
    start_time: float,
    elapsed_time: float,
    first_token_time: float,
    context_len: Optional[int],
    max_seq_len: int,
):
    prompt_processing_time = first_token_time - start_time

    initial_response = (
        f"Metrics: {generated_tokens} T generated in "
        f"{round(elapsed_time, 2)} seconds"
    )
    itemization = []
    extra_parts = []

    # Add prompt tokens per second
    prompt_tokens_per_second = (
        "Indeterminate"
        if elapsed_time == 0
        else round(context_len / prompt_processing_time, 2)
    )
    itemization.append(f"PP: {prompt_tokens_per_second} T/s")

    # Add tokens per second
    tokens_per_second = (
        "Indeterminate"
        if elapsed_time == 0
        else round(generated_tokens / elapsed_time, 2)
    )
    itemization.append(f"GEN: {tokens_per_second} T/s")

    # Add context (original token count)
    if context_len:
        itemization.append(f"context {context_len} T")

        if context_len > max_seq_len:
            extra_parts.append("<-- Not accurate (truncated)")

    # Print output
    logger.info(
        initial_response + " (" + ", ".join(itemization) + ") " + " ".join(extra_parts)
    )
