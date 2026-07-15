import pathlib
from common import model
from endpoints.OAI.types.common import UsageStats
from common.tabby_config import config
from common.auth import get_key_permission
from common.logger import xlogger
from common.networking import handle_request_error
from fastapi import HTTPException, Request


def get_usage_stats(
    generation: dict,
) -> UsageStats | None:
    """
    Collect usage stats from generation if it is a finish chunk
    """
    if "finish_reason" not in generation:
        return None

    prompt_tokens = generation.get("prompt_tokens", 0)
    completion_tokens = generation.get("gen_tokens", 0)
    usage_stats = UsageStats(
        prompt_tokens=prompt_tokens,
        prompt_time=generation.get("prompt_time"),
        prompt_tokens_per_sec=generation.get("prompt_tokens_per_sec"),
        completion_tokens=completion_tokens,
        completion_time=generation.get("gen_time"),
        completion_tokens_per_sec=generation.get("gen_tokens_per_sec"),
        total_tokens=prompt_tokens + completion_tokens,
        total_time=generation.get("total_time"),
    )
    return usage_stats


def aggregate_usage_stats(usage_stats_list: list[UsageStats]) -> UsageStats:
    if len(usage_stats_list) == 1:
        return usage_stats_list[0]

    usl = usage_stats_list
    prompt_tokens = usl[0].prompt_tokens
    prompt_time = usl[0].prompt_time
    prompt_tokens_per_sec = usl[0].prompt_tokens_per_sec
    completion_tokens = sum(us.completion_tokens for us in usl)
    completion_time = max(us.completion_time for us in usl)
    completion_tokens_per_sec = completion_tokens / (completion_time + 1e-20)
    total_tokens = prompt_tokens + completion_tokens
    total_time = prompt_time + completion_time

    usage_stats = UsageStats(
        prompt_tokens=prompt_tokens,
        prompt_time=prompt_time,
        prompt_tokens_per_sec=prompt_tokens_per_sec,
        completion_tokens=completion_tokens,
        completion_time=completion_time,
        completion_tokens_per_sec=completion_tokens_per_sec,
        total_tokens=total_tokens,
        total_time=total_time,
    )
    return usage_stats


async def load_inline_model(model_name: str, request: Request):
    """Load a model from the data.model parameter"""

    # Return if the model container already exists and the model is fully loaded
    if model.container and model.container.model_dir.name == model_name and model.container.loaded:
        return

    # Return if inline loading is disabled
    # Also warn if an admin key is used
    if not config.model.inline_model_loading:
        if get_key_permission(request) == "admin":
            xlogger.warning(
                f"Unable to switch model to {model_name} because "
                '"inline_model_loading" is not True in config.yml.'
            )

        return

    is_dummy_model = config.model.use_dummy_models and model_name in config.model.dummy_model_names

    # Error if an invalid key is passed
    # If a dummy model is provided, don't error
    if get_key_permission(request) != "admin":
        if not is_dummy_model:
            error_message = handle_request_error(
                f"Unable to switch model to {model_name} because " + "an admin key isn't provided",
                exc_info=False,
            ).error.message

            raise HTTPException(401, error_message)
        else:
            return

    # Start inline loading
    # Past here, user is assumed to be admin

    # Skip if the model is a dummy
    if is_dummy_model:
        xlogger.warning(f"Dummy model {str(model_name)} provided. Skipping inline load.")
        return

    model_path = pathlib.Path(config.model.model_dir)
    model_path = model_path / model_name

    # Model path doesn't exist
    if not model_path.exists():
        xlogger.warning(f"Could not find model path {str(model_path)}. Skipping inline model load.")

        return

    # Load the model and also add draft dir
    await model.load_model(
        model_path,
        draft_model=config.draft_model.model_dump(include={"draft_model_dir"}),
    )
