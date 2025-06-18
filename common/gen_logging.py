"""
Functions for logging generation events.
"""

from loguru import logger
from typing import Optional

from common.tabby_config import config


def broadcast_status():
    """Broadcasts the current logging status"""
    enabled = []
    if config.logging.log_prompt:
        enabled.append("prompts")

    if config.logging.log_generation_params:
        enabled.append("generation params")

    if len(enabled) > 0:
        logger.info("Generation logging is enabled for: " + ", ".join(enabled))
    else:
        logger.info("Generation logging is disabled")


def log_generation_params(**kwargs):
    """Logs generation parameters to console."""
    if config.logging.log_generation_params:
        logger.info(f"Generation options: {kwargs}\n")


def log_prompt(prompt: str, request_id: str, negative_prompt: Optional[str] = None):
    """Logs the prompt to console."""
    if config.logging.log_prompt:
        formatted_prompt = "\n" + prompt
        logger.info(
            f"Prompt (ID: {request_id}): {formatted_prompt if prompt else 'Empty'}\n"
        )

        if negative_prompt:
            formatted_negative_prompt = "\n" + negative_prompt
            logger.info(f"Negative Prompt: {formatted_negative_prompt}\n")


def log_response(request_id: str, response: str):
    """Logs the response to console."""
    if config.logging.log_prompt:
        formatted_response = "\n" + response
        logger.info(
            f"Response (ID: {request_id}): "
            f"{formatted_response if response else 'Empty'}\n"
        )


def log_metrics(
    request_id: str,
    metrics: dict,
    context_len: Optional[int],
    max_seq_len: int,
):
    initial_response = (
        f"Metrics (ID: {request_id}): {metrics.get('gen_tokens')} "
        f"tokens generated in {metrics.get('total_time')} seconds"
    )
    itemization = []
    extra_parts = []

    itemization.append(f"Queue: {metrics.get('queue_time')} s")

    cached_tokens = metrics.get("cached_tokens")
    prompt_tokens = metrics.get("prompt_tokens")

    itemization.append(
        f"Process: {cached_tokens} cached tokens and "
        f"{prompt_tokens - cached_tokens} new tokens at "
        f"{metrics.get('prompt_tokens_per_sec')} T/s"
    )

    itemization.append(f"Generate: {metrics.get('gen_tokens_per_sec')} T/s")

    # Add context (original token count)
    if context_len:
        itemization.append(f"Context: {context_len} tokens")

        if context_len > max_seq_len:
            extra_parts.append("<-- Not accurate (truncated)")

    # Print output
    logger.info(
        initial_response + " (" + ", ".join(itemization) + ") " + " ".join(extra_parts)
    )
