"""
Functions for logging generation events.
"""

from pydantic import BaseModel
from loguru import logger
from typing import Dict, Optional

from common.tabby_config import config

# Global logging preferences constant
PREFERENCES = config.logging


def broadcast_status():
    """Broadcasts the current logging status"""
    enabled = []
    if PREFERENCES.log_prompt:
        enabled.append("prompts")

    if PREFERENCES.log_generation_params:
        enabled.append("generation params")

    if len(enabled) > 0:
        logger.info("Generation logging is enabled for: " + ", ".join(enabled))
    else:
        logger.info("Generation logging is disabled")


def log_generation_params(**kwargs):
    """Logs generation parameters to console."""
    if PREFERENCES.log_generation_params:
        logger.info(f"Generation options: {kwargs}\n")


def log_prompt(prompt: str, request_id: str, negative_prompt: Optional[str]):
    """Logs the prompt to console."""
    if PREFERENCES.log_prompt:
        formatted_prompt = "\n" + prompt
        logger.info(
            f"Prompt (ID: {request_id}): {formatted_prompt if prompt else 'Empty'}\n"
        )

        if negative_prompt:
            formatted_negative_prompt = "\n" + negative_prompt
            logger.info(f"Negative Prompt: {formatted_negative_prompt}\n")


def log_response(request_id: str, response: str):
    """Logs the response to console."""
    if PREFERENCES.log_prompt:
        formatted_response = "\n" + response
        logger.info(
            f"Response (ID: {request_id}): "
            f"{formatted_response if response else 'Empty'}\n"
        )


def log_metrics(
    request_id: str,
    queue_time: float,
    prompt_tokens: int,
    cached_tokens: int,
    prompt_time: float,
    generated_tokens: int,
    generate_time: float,
    context_len: Optional[int],
    max_seq_len: int,
):
    initial_response = (
        f"Metrics (ID: {request_id}): {generated_tokens} tokens generated in "
        f"{round(queue_time + prompt_time + generate_time, 2)} seconds"
    )
    itemization = []
    extra_parts = []

    itemization.append(f"Queue: {round(queue_time, 2)} s")

    prompt_ts = (
        "Indeterminate"
        if prompt_time == 0
        else round((prompt_tokens - cached_tokens) / prompt_time, 2)
    )
    itemization.append(
        f"Process: {cached_tokens} cached tokens and "
        f"{prompt_tokens - cached_tokens} new tokens at {prompt_ts} T/s"
    )

    generate_ts = (
        "Indeterminate"
        if generate_time == 0
        else round(generated_tokens / generate_time, 2)
    )
    itemization.append(f"Generate: {generate_ts} T/s")

    # Add context (original token count)
    if context_len:
        itemization.append(f"Context: {context_len} tokens")

        if context_len > max_seq_len:
            extra_parts.append("<-- Not accurate (truncated)")

    # Print output
    logger.info(
        initial_response + " (" + ", ".join(itemization) + ") " + " ".join(extra_parts)
    )
